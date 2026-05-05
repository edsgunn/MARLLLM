"""
vLLM-based sampling engine for population RL training.

Wraps ``vllm.LLM`` with:
  * multi-LoRA support keyed by population agent name
  * on-disk LoRA hot-swap weight sync (called after each ``optimizer.step()``)
  * an interface returning ``(token_ids, logprobs)`` matching the contract
    of ``Agent.act_batch`` so the trainer can use it as a drop-in replacement.

Phase 1 limitations
-------------------
* Only the ``--lora-shared-base`` training path is supported. Full
  fine-tuning would need vLLM's ``update_weights`` API rather than
  ``add_lora``; that's a separate piece of work.
* Single-process / single-engine. Memory is shared with the training model
  via ``gpu_memory_utilization``. You will likely need
  ``--gradient-checkpointing`` for both to fit.
* Concordia GM still uses the HF backbone — vLLM is rollout-only.

Imports vllm lazily so the rest of the codebase still works without it.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y", "on")


class VLLMSamplingEngine:
    """Sampling engine backed by vllm.LLM with LoRA hot-swap.

    Construction loads the base model into vLLM and writes initial LoRA
    weights to disk. ``sync_all_adapters`` is called after each optimizer
    step to flush updated weights and bump the cache id so the next
    ``generate`` reads fresh weights.

    The integer LoRA ids handed to vLLM are monotonically increasing so a
    fresh id always invalidates the cached copy. Old paths are kept on disk
    until ``close`` (deleting them while the engine still references the
    id is racy — keep them cheap by writing into ``/dev/shm`` if possible).
    """

    def __init__(
        self,
        model_name: str,
        peft_model: Any,
        adapter_names: list[str],
        *,
        max_lora_rank: int,
        gpu_memory_utilization: float = 0.45,
        max_model_len: int | None = None,
        dtype: str = "bfloat16",
        adapter_dir: str | None = None,
        enforce_eager: bool = False,
        max_num_seqs: int | None = None,
        quantization: str | None = None,
        local_rank: int = 0,
    ) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VLLMSamplingEngine requires the `vllm` package: uv add vllm"
            ) from e

        if not adapter_names:
            raise ValueError("adapter_names must be non-empty")

        self._model_name = model_name
        self._adapter_names = list(adapter_names)
        self._global_id_counter = 0
        self._current_ids: dict[str, int] = {}
        self._current_paths: dict[str, Path] = {}
        # Parent of each adapter's current `id{N}` dir. Tracked so we can
        # rm -rf the previous version when a new sync lands; otherwise the
        # staging dir grows by ~(num_adapters * adapter_size) every step
        # and eventually fills /dev/shm.
        self._current_id_dirs: dict[str, Path] = {}

        # Prefer /dev/shm for adapter staging (tmpfs → fast write+read).
        if adapter_dir is None:
            shm = Path("/dev/shm")
            base = shm if shm.is_dir() and os.access(shm, os.W_OK) else None
            # Best-effort sweep of stale staging dirs from prior crashed jobs
            # on the same node — mkdtemp doesn't clean them up, and a packed
            # /dev/shm causes the very next save_pretrained to ENOSPC.
            if base is not None:
                self._sweep_stale_staging_dirs(base)
            self._adapter_root = Path(
                tempfile.mkdtemp(prefix="marlllm_lora_", dir=str(base) if base else None)
            )
        else:
            self._adapter_root = Path(adapter_dir)
            self._adapter_root.mkdir(parents=True, exist_ok=True)

        logger.info("vLLM adapter staging dir: %s", self._adapter_root)

        # Save initial weights for every adapter before constructing the engine.
        for name in self._adapter_names:
            self._save_adapter(peft_model, name)

        from vllm import LLM
        llm_kwargs: dict[str, Any] = dict(
            model=model_name,
            enable_lora=True,
            max_loras=len(self._adapter_names),
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if quantization is not None:
            llm_kwargs["quantization"] = quantization

        # Construct vLLM in an isolated env so the spawned EngineCore subprocess
        # is independent of the trainer's distributed setup:
        #   * CUDA_VISIBLE_DEVICES → pin to this rank's GPU (else every engine
        #     collides on cuda:0 and OOMs).
        #   * WORLD_SIZE / RANK / LOCAL_RANK / MASTER_ADDR / MASTER_PORT (set
        #     by torchrun) → must be cleared. Otherwise the EngineCore inherits
        #     them and torch.distributed.init_process_group inside the engine
        #     waits for `WORLD_SIZE-1` peers to join its private TCPStore.
        #     They never do → 600s timeout per engine, no progress.
        # Env is restored after LLM() so the trainer's DDP context is intact.
        _env_overrides = {
            "CUDA_VISIBLE_DEVICES": str(local_rank),
            "WORLD_SIZE": None,
            "RANK": None,
            "LOCAL_RANK": None,
            "LOCAL_WORLD_SIZE": None,
            "MASTER_ADDR": None,
            "MASTER_PORT": None,
            "GROUP_RANK": None,
            "ROLE_RANK": None,
            "ROLE_WORLD_SIZE": None,
            "TORCHELASTIC_RUN_ID": None,
            "TORCHELASTIC_RESTART_COUNT": None,
            "TORCHELASTIC_MAX_RESTARTS": None,
            "TORCHELASTIC_USE_AGENT_STORE": None,
        }
        _saved_env = {k: os.environ.get(k) for k in _env_overrides}
        for k, v in _env_overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            logger.info(
                "Constructing vllm.LLM on local_rank=%d "
                "(CUDA_VISIBLE_DEVICES=%s, torchrun env scrubbed): %s",
                local_rank, os.environ.get("CUDA_VISIBLE_DEVICES"), llm_kwargs,
            )
            self._llm = LLM(**llm_kwargs)
        finally:
            for k, v in _saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # ------------------------------------------------------------------ #
    # Adapter management                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sweep_stale_staging_dirs(base: Path) -> None:
        """Remove orphaned ``marlllm_lora_*`` dirs on this node.

        Compute nodes are reused across SLURM jobs; if a previous run
        crashed mid-training the staging dir leaks (mkdtemp doesn't unlink
        on exit). On a packed /dev/shm the next job ENOSPCs at startup.

        Only touch directories that are (a) owned by the current uid and
        (b) untouched for at least an hour — so we never delete a sibling
        rank's freshly-mkdtemp'd dir or a concurrent job's active state.
        """
        import time
        try:
            uid = os.getuid()
            cutoff = time.time() - 3600
            for entry in base.glob("marlllm_lora_*"):
                try:
                    if not entry.is_dir():
                        continue
                    st = entry.stat()
                    if st.st_uid != uid or st.st_mtime > cutoff:
                        continue
                    shutil.rmtree(entry, ignore_errors=True)
                except OSError:
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("stale staging dir sweep failed: %s", exc)

    def _save_adapter(self, peft_model: Any, name: str) -> Path:
        """Save the named adapter to a fresh path and bump its cache id.

        peft.save_pretrained writes the *currently active* adapter, so we
        set_adapter() around the call and restore the previous active one
        afterwards.
        """
        prev_active = getattr(peft_model, "active_adapter", None)
        try:
            peft_model.set_adapter(name)
            self._global_id_counter += 1
            int_id = self._global_id_counter
            path = self._adapter_root / name / f"id{int_id}"
            path.mkdir(parents=True, exist_ok=True)
            try:
                peft_model.save_pretrained(
                    str(path), selected_adapters=[name]
                )
            except TypeError:
                # Older peft: save_pretrained doesn't take selected_adapters.
                peft_model.save_pretrained(str(path))

            # PEFT's save_pretrained writes any non-"default" named adapter
            # into a `<save_dir>/<adapter_name>/` subdirectory, not directly
            # into save_dir. vLLM's LoRARequest.lora_path must point at the
            # directory that actually contains adapter_config.json, so resolve
            # it explicitly here. Fall back to `path` for the "default" case.
            actual_path = path / name
            if not (actual_path / "adapter_config.json").is_file():
                if (path / "adapter_config.json").is_file():
                    actual_path = path
                else:
                    raise FileNotFoundError(
                        f"adapter_config.json not found after save_pretrained "
                        f"under {path} (checked {path}/ and {actual_path}/). "
                        f"Contents: {list(path.rglob('*'))[:20]}"
                    )
            prev_id_dir = self._current_id_dirs.get(name)
            self._current_ids[name] = int_id
            self._current_paths[name] = actual_path
            self._current_id_dirs[name] = path
            # vLLM caches LoRAs by lora_int_id and we only ever hand it the
            # newest id, so the previous on-disk copy is unreferenced and
            # safe to drop. Without this the staging dir grows unbounded.
            if prev_id_dir is not None and prev_id_dir != path:
                shutil.rmtree(prev_id_dir, ignore_errors=True)
        finally:
            if prev_active is not None and prev_active != name:
                try:
                    peft_model.set_adapter(prev_active)
                except Exception:
                    pass
        return self._current_paths[name]

    def sync_all_adapters(self, peft_model: Any) -> None:
        """Flush every adapter's current weights to disk with a new id.

        Call this once per training step, after ``optimizer.step()``. The
        next ``generate`` for an adapter will pick up the fresh weights.
        """
        for name in self._adapter_names:
            self._save_adapter(peft_model, name)

    def _lora_request(self, name: str):
        from vllm.lora.request import LoRARequest
        return LoRARequest(
            lora_name=name,
            lora_int_id=self._current_ids[name],
            lora_path=str(self._current_paths[name]),
        )

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        contexts: list[list[int]],
        n_tokens: int,
        temperature: float,
        eos_token_ids: list[int] | None,
        adapter_name: str,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Drop-in replacement for ``Agent.act_batch``.

        Returns (token_ids_per_seq, logprobs_per_seq). The logprob for each
        emitted token is the log-probability of *that token* under the
        sampling distribution at that step (after temperature scaling),
        which matches the contract of the existing HF act_batch path.
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sp = SamplingParams(
            temperature=float(temperature) if temperature > 0 else 0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=int(n_tokens),
            stop_token_ids=list(eos_token_ids) if eos_token_ids else None,
            logprobs=0,  # 0 → only the chosen token's logprob is returned
            ignore_eos=False,
        )

        prompts = [TokensPrompt(prompt_token_ids=list(c)) for c in contexts]
        outputs = self._llm.generate(
            prompts,
            sampling_params=sp,
            lora_request=self._lora_request(adapter_name),
            use_tqdm=False,
        )

        # vLLM is not guaranteed to return outputs in the same order as
        # inputs; re-order via request_id if exposed, otherwise rely on the
        # documented behaviour that order is preserved when prompts is a list.
        all_ids: list[list[int]] = []
        all_lps: list[list[float]] = []
        for out in outputs:
            comp = out.outputs[0]
            ids = list(comp.token_ids)
            lps_dicts = list(comp.logprobs or [])
            lps: list[float] = []
            for tok, lpd in zip(ids, lps_dicts):
                if lpd is None:
                    lps.append(0.0)
                    continue
                # lpd: dict[token_id, vllm.Logprob | float]
                v = lpd.get(tok)
                if v is None:
                    lps.append(0.0)
                else:
                    lps.append(float(getattr(v, "logprob", v)))
            all_ids.append(ids)
            all_lps.append(lps)
        return all_ids, all_lps

    # ------------------------------------------------------------------ #
    # Sleep / wake (memory ↔ CPU) so the trainer can use the GPU         #
    # ------------------------------------------------------------------ #

    def sleep(self, level: int = 2) -> None:
        """Offload vLLM weights (level=1) or weights+KV cache (level=2) to CPU.

        Call between rollouts to free GPU memory for the loss/backward step.
        Idempotent — calling sleep() on an already-asleep engine is a no-op.
        """
        if not getattr(self, "_is_asleep", False):
            self._llm.sleep(level=level)
            self._is_asleep = True

    def wake_up(self) -> None:
        """Restore vLLM weights and KV cache to GPU. Idempotent."""
        if getattr(self, "_is_asleep", False):
            self._llm.wake_up()
            self._is_asleep = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Best-effort cleanup of staged adapter directories."""
        try:
            shutil.rmtree(self._adapter_root, ignore_errors=True)
        except Exception:
            pass
