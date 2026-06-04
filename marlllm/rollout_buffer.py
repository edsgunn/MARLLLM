"""Async rollout buffer + worker thread.

Background
----------
Synchronous rollout/train couples the learner to the slowest episode of every
iteration (`time/rollout_s` is 80-90% of iter wall-clock in our measured
runs). The fix is to disaggregate generation and training: rollout workers
write trajectories into a buffer; the learner consumes whatever is ready.

The CCSM-specific subtlety (see ``A Guide to Training Agentic LLMs on Off-
Policy Surprise Minimisation.md``) is that the "reward" *is* the current
model's surprise, so:

  * Returns/advantages MUST be recomputed under current θ at consumption
    time — we already do this in ``compute_loss`` / ``compute_loss_chunked``,
    no change needed.
  * The trajectory distribution itself becomes stale: action tokens were
    sampled under θ_old. The off-policy correction is per-action-token
    PPO-clip importance weighting, applied at loss time. The behaviour
    log-probs are already stored in ``RolloutBatch.act_log_probs_old``.
  * Perception (NTP on OBS tokens) has no importance-weighting story; the
    guide recommends keeping replay shallow (small staleness budget). This
    module exposes ``max_staleness`` to enforce that bound.

Design
------
* One worker thread per rank. Runs in the same process as the learner; both
  share the same ``peft_model`` and ``sampling_engine``. No multiprocessing.
* The worker calls ``_collect_episodes_batched`` (heavy method on the
  trainer). Each call produces one "iter's worth" of raw episodes. These
  are stamped with the current ``policy_version`` and queued.
* Adapter weight sync happens in the *training* thread immediately after
  ``optimizer.step()``: writes new LoRA weights to disk and bumps vLLM's
  internal ``_current_ids[name]``. In-flight ``vllm.LLM.generate`` calls
  in the worker hold a ``LoRARequest`` constructed earlier and continue
  using the old weights — that's the source of staleness, and it's a
  feature, not a race. The very next ``generate`` after the sync picks up
  the new lora_int_id.
* DDP: each rank runs its own buffer + worker independently. Cross-rank
  synchronisation is at the existing all-reduce of gradients — that's the
  one barrier all ranks hit together. If one rank's buffer drains it
  blocks on ``buffer.get()``, holding up the all-reduce, but in practice
  every rank's rollout takes about the same time (same model, env mix,
  prng), so this averages out.

Constraints
-----------
The buffer is bounded by ``max_size`` (in iter-batches). When full, the
worker blocks on the put. So the worker is at most ``max_size`` iters
ahead of the trainer. With ``max_staleness=max_size`` you can never see
a trajectory older than that.

Failure handling
----------------
If the worker raises, it stores the exception on ``self.exception`` and
exits. The trainer detects this on the next ``get`` (signalled via
sentinel) and re-raises, so a worker crash surfaces as a normal-looking
exception in the main thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class _BufferItem:
    """One iter-batch of raw episodes plus the policy version it was sampled under."""
    episodes: list           # whatever ``_collect_episodes_batched`` returns (list of _RawEpisode)
    rollout_stats: dict      # the worker's per-batch timing/throughput stats
    policy_version: int      # version when episodes were *finished* (after worker re-loaded weights)


class RolloutBuffer:
    """Thread-safe bounded FIFO of iter-batches with policy-version stamps.

    Producer (rollout worker) calls ``put``. Consumer (trainer) calls
    ``get``. Staleness filtering happens on the consumer side using
    ``policy_version``.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._q: queue.Queue[_BufferItem | None] = queue.Queue(maxsize=max_size)
        self._sentinel = None  # signals shutdown / worker death

    def put(self, item: _BufferItem) -> None:
        """Block until there is room, then enqueue."""
        self._q.put(item)

    def put_sentinel(self) -> None:
        """Signal end of stream / worker error."""
        self._q.put(self._sentinel)

    def get(self, timeout: float | None = None) -> _BufferItem:
        """Block until an item is available, or raise ``BufferClosed``."""
        item = self._q.get(timeout=timeout)
        if item is None:
            raise BufferClosed("rollout worker has stopped")
        return item

    @property
    def size(self) -> int:
        return self._q.qsize()

    @property
    def maxsize(self) -> int:
        return self._q.maxsize


class BufferClosed(RuntimeError):
    """Raised by ``RolloutBuffer.get`` after the worker has stopped."""


class RolloutWorker:
    """Thread that calls ``produce_one_batch`` in a loop and feeds the buffer.

    ``produce_one_batch`` is a callable provided by the trainer (typically a
    bound method on ``PopulationTrainer`` that wraps
    ``_collect_episodes_batched`` plus the worker-side adapter refresh).
    ``current_policy_version`` is a callable returning the latest version
    the trainer has published; the worker reads it just before each batch
    so the stamp reflects the weights actually used.

    Lifecycle: construct → ``start()`` → ``stop()`` (or trainer exit).
    Exceptions from inside the worker are stored on ``self.exception`` and
    surfaced via the buffer sentinel; the next ``buffer.get`` raises
    ``BufferClosed`` and the trainer can inspect ``worker.exception``.
    """

    def __init__(
        self,
        buffer: RolloutBuffer,
        produce_one_batch: Callable[[int], tuple[list, dict]],
        current_policy_version: Callable[[], int],
        on_version_advance: Callable[[], None] | None = None,
        name: str = "rollout-worker",
    ) -> None:
        self._buffer = buffer
        self._produce_one_batch = produce_one_batch
        self._current_policy_version = current_policy_version
        self._on_version_advance = on_version_advance
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.exception: BaseException | None = None
        self._last_seen_version = -1

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float | None = 30.0) -> None:
        """Signal the worker to exit at the next batch boundary."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                # Pick up newly-published adapter weights before the next batch.
                # The trainer thread writes them after each optimizer.step();
                # we read the version atomically and let the hook refresh
                # vLLM's view if needed. The "policy version stamped on this
                # batch" is the version *at the start of generation* — which
                # is what the action log-probs were sampled under.
                v = self._current_policy_version()
                if v != self._last_seen_version:
                    if self._on_version_advance is not None:
                        try:
                            self._on_version_advance()
                        except Exception:
                            logger.exception(
                                "rollout worker: on_version_advance failed"
                            )
                            raise
                    self._last_seen_version = v
                if self._stop_event.is_set():
                    break
                t0 = time.perf_counter()
                episodes, stats = self._produce_one_batch(v)
                stats = dict(stats)
                stats["worker_batch_s"] = time.perf_counter() - t0
                self._buffer.put(_BufferItem(
                    episodes=episodes,
                    rollout_stats=stats,
                    policy_version=v,
                ))
        except BaseException as e:  # noqa: BLE001 — surface to main thread
            self.exception = e
            logger.exception("rollout worker died with exception")
        finally:
            self._buffer.put_sentinel()
