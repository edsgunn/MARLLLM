"""`extract` — run the judge over a corpus slice; write event records.

Respects the cache, resumes from already-extracted posts, runs judge calls
with a bounded thread pool, and prints a usage/cost tally at the end.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .adapter import iter_corpus
from .backends import PRICING, make_judge_backend
from .cache import Cache
from .judge import Judge
from .persistence import EventStore


def run_extract(cfg, substrates=None, run_ids=None, iter_min=None, iter_max=None, limit=None):
    cfg.make_dirs()
    backend = make_judge_backend(cfg.judge)
    cache = Cache(cfg.paths.cache_db)
    judge = Judge(backend, cache, cfg.judge)
    store = EventStore(cfg.paths.events)

    done = store.existing_keys()
    if done:
        print(f"[extract] {len(done)} posts already extracted; resuming.")

    # Build the work list: (episode, candidate_idx). Every post is a candidate.
    work = []
    for ep in iter_corpus(cfg.corpus, substrates, run_ids, iter_min, iter_max):
        for idx, post in enumerate(ep.posts):
            key = (post.run_id, post.iteration, ep.episode_id, post.turn_index)
            if key in done:
                continue
            work.append((ep.posts, idx))
            if limit and len(work) >= limit:
                break
        if limit and len(work) >= limit:
            break

    total = len(work)
    print(f"[extract] {total} posts to label "
          f"(provider={cfg.judge.provider} model={cfg.judge.model} window_k={cfg.judge.window_k}).")
    if total == 0:
        print("[extract] nothing to do.")
        cache.close()
        return

    # Preflight: label the first post synchronously so a bad key / config
    # fails fast with one clean message instead of a 6k-future traceback storm.
    n_done = 0
    try:
        first_posts, first_idx = work[0]
        store.append(judge.label_post(first_posts, first_idx))
        n_done = 1
    except Exception as e:  # noqa: BLE001 — surface a friendly hint, then re-raise
        cache.close()
        _explain_failure(e)
        raise

    # A lock-free pattern: workers return records; the main thread writes them
    # to the append-only store (keeps JSONL writes single-threaded).
    with ThreadPoolExecutor(max_workers=cfg.judge.concurrency) as ex:
        futures = {ex.submit(judge.label_post, posts, idx): None for posts, idx in work[1:]}
        for fut in as_completed(futures):
            rec = fut.result()
            store.append(rec)
            n_done += 1
            if n_done % 50 == 0 or n_done == total:
                print(f"[extract] {n_done}/{total} "
                      f"(api_calls={judge.calls} cache_hits={judge.cache_hits} "
                      f"parse_failures={judge.parse_failures})", flush=True)

    cache.close()
    _print_cost(judge)


def _explain_failure(e: Exception):
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    if status == 401 or name == "AuthenticationError":
        print(
            "\n[extract] ABORTED on the first call: the API key was rejected (401).\n"
            "  The work list is fine — this is purely auth. Check, in the SAME shell:\n"
            "    echo -n \"$ANTHROPIC_API_KEY\" | wc -c     # expect ~100+ chars, no trailing newline\n"
            "    printf '%s' \"$ANTHROPIC_API_KEY\" | head -c 7   # should print 'sk-ant-'\n"
            "  Common causes: a trailing space/newline, quotes captured into the value,\n"
            "  `export VAR = x` (spaces around =), an OpenAI key in ANTHROPIC_API_KEY,\n"
            "  or a key scoped to a different workspace/org. Re-export and retry —\n"
            "  nothing was billed and `extract` resumes from where it left off.",
            flush=True,
        )
    else:
        print(f"\n[extract] ABORTED on the first call: {name}: {e}", flush=True)


def _print_cost(judge):
    print("\n=== extract usage ===")
    print(f"new API calls : {judge.calls}")
    print(f"cache hits    : {judge.cache_hits}")
    print(f"parse failures: {judge.parse_failures}")
    print(f"input tokens  : {judge.usage_in:,}")
    print(f"output tokens : {judge.usage_out:,}")
    price = PRICING.get(judge.cfg.model)
    if price:
        cin, cout = price
        cost = judge.usage_in / 1e6 * cin + judge.usage_out / 1e6 * cout
        print(f"est. cost     : ${cost:,.2f}  (model {judge.cfg.model}; new calls only)")
    else:
        print(f"est. cost     : (no pricing entry for {judge.cfg.model})")
