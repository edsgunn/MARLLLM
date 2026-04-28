#!/usr/bin/env python3
"""
Negotiation trace analysis script.
Analyses decoded text from agent_contexts and environment_log fields.
Does NOT use the tokenizer.
"""

import json
import os
import re
import glob
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

RUNS_ROOT = "/lus/lfs1aip2/projects/a5l/egunn/projects/MARLLLM/runs"

# ── helpers ──────────────────────────────────────────────────────────────────

ALLOCATION_RE = re.compile(
    r'books\s*=\s*(\d+)\s+hats\s*=\s*(\d+)\s+balls\s*=\s*(\d+)',
    re.IGNORECASE
)
SPECIAL_TOKEN_RE = re.compile(r'<\|[^|>]+\|>')
DEGENERATE_RE    = re.compile(r'(Human:|Assistant:|User:|System:)', re.IGNORECASE)

def non_ascii_fraction(text):
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)

def detect_scripts(text):
    """Return set of Unicode script categories (block names simplified)."""
    scripts = set()
    for ch in text:
        if ord(ch) > 127:
            try:
                name = unicodedata.name(ch, '')
                # Extract leading script word
                parts = name.split()
                if parts:
                    scripts.add(parts[0])
            except Exception:
                pass
    return scripts

def is_truncated(text):
    """
    Heuristic: action text looks truncated if:
    - it ends mid-word (last char is alphanumeric/underscore, not whitespace/punctuation)
    - or it ends mid-number with no terminating whitespace
    Does NOT count pure-whitespace strings as truncated.
    """
    stripped = text.rstrip('\n\r\t ')
    if not stripped:
        return False  # empty / all whitespace — collapsed, not truncated
    last = stripped[-1]
    # truncated if last real character is a word character (not punctuation/space)
    return last.isalnum() or last in ('_', '-', '=')

def is_collapsed(text):
    """Entirely whitespace (\\n spam)."""
    return text.strip() == ''

def has_allocation(text):
    return bool(ALLOCATION_RE.search(text))

def extract_allocations(text):
    return ALLOCATION_RE.findall(text)

def has_special_token(text):
    return bool(SPECIAL_TOKEN_RE.search(text))

def has_degenerate_roleplay(text):
    return bool(DEGENERATE_RE.search(text))

def char_length(text):
    return len(text)

# ── per-trace analysis ───────────────────────────────────────────────────────

def analyse_trace(path):
    with open(path) as f:
        d = json.load(f)

    meta = d['meta']
    env_log = d.get('environment_log', [])
    agent_ctxs = d.get('agent_contexts', {})

    # Collect action entries from environment_log
    actions = [e for e in env_log if e.get('type') == 'act']

    # Collect action entries from agent_contexts (type == 'act')
    ctx_actions = []
    for agent, entries in agent_ctxs.items():
        for e in entries:
            if e.get('type') == 'act':
                ctx_actions.append({'agent': agent, 'text': e['text']})

    # Use env_log actions as primary (they include agent label)
    all_act_texts = [e['text'] for e in actions]

    # ── outcome ──
    outcome = meta.get('outcome', 'unknown')
    score   = meta.get('score', 0)
    items   = meta.get('items', [])

    # ── last action (final submission) ──
    # In env_log the final submissions are the last two 'act' entries
    # (one per agent, before the final obs)
    submission_texts = all_act_texts[-2:] if len(all_act_texts) >= 2 else all_act_texts

    # ── token truncation ──
    n_truncated  = sum(1 for t in all_act_texts if is_truncated(t))
    n_collapsed  = sum(1 for t in all_act_texts if is_collapsed(t))
    lengths      = [char_length(t) for t in all_act_texts]
    avg_len      = sum(lengths) / len(lengths) if lengths else 0

    # ── language drift ──
    all_text = '\n'.join(all_act_texts)
    na_frac  = non_ascii_fraction(all_text)
    scripts  = detect_scripts(all_text)

    # ── proposal coherence ──
    n_has_alloc    = sum(1 for t in all_act_texts if has_allocation(t))
    n_no_alloc     = len(all_act_texts) - n_has_alloc
    sub_allocs     = [extract_allocations(t) for t in submission_texts]
    sub_has_alloc  = [bool(a) for a in sub_allocs]

    # ── dialogue quality ──
    n_special_tok  = sum(1 for t in all_act_texts if has_special_token(t))
    n_roleplay     = sum(1 for t in all_act_texts if has_degenerate_roleplay(t))
    special_examples = [t[:200] for t in all_act_texts if has_special_token(t)][:2]

    # ── agreement failure mode ──
    failure_mode = 'n/a'
    if outcome == 'no_deal':
        both_parse = all(sub_has_alloc)
        none_parse = not any(sub_has_alloc)
        if none_parse:
            failure_mode = 'both_parse_fail'
        elif not both_parse:
            failure_mode = 'one_parse_fail'
        else:
            # Both parsed — check if allocations sum correctly
            try:
                alloc_vals = []
                for allocs in sub_allocs:
                    if allocs:
                        alloc_vals.append(tuple(int(x) for x in allocs[0]))
                if len(alloc_vals) == 2:
                    totals = [a+b for a,b in zip(alloc_vals[0], alloc_vals[1])]
                    if totals == list(items):
                        failure_mode = 'parse_ok_sum_ok_but_no_deal'  # shouldn't happen
                    else:
                        failure_mode = 'parse_ok_sum_wrong'
                else:
                    failure_mode = 'parse_ok_but_mismatch'
            except Exception:
                failure_mode = 'parse_ok_analysis_error'

    # ── examples ──
    example_actions = all_act_texts[:3]  # first 3 actions

    return {
        'path': path,
        'iteration': meta.get('iteration', 0),
        'outcome': outcome,
        'score': score,
        'items': items,
        'n_actions': len(all_act_texts),
        'n_truncated': n_truncated,
        'n_collapsed': n_collapsed,
        'avg_len': avg_len,
        'lengths': lengths,
        'na_frac': na_frac,
        'scripts': scripts,
        'n_has_alloc': n_has_alloc,
        'n_no_alloc': n_no_alloc,
        'sub_has_alloc': sub_has_alloc,
        'n_special_tok': n_special_tok,
        'n_roleplay': n_roleplay,
        'special_examples': special_examples,
        'failure_mode': failure_mode,
        'example_actions': example_actions,
        'submission_texts': submission_texts,
    }

# ── run-level analysis ────────────────────────────────────────────────────────

def analyse_run(run_dir, sample_iters=None):
    """Analyse a run directory. If sample_iters given, only load those files."""
    trace_files = sorted(glob.glob(f'{run_dir}/traces/iter_*.json'))
    if not trace_files:
        return None

    if sample_iters is not None:
        # Pick closest available iter to each requested iter
        available = {}
        for f in trace_files:
            m = re.search(r'iter_(\d+)\.json$', f)
            if m:
                available[int(m.group(1))] = f
        selected = []
        for target in sample_iters:
            if not available:
                break
            closest = min(available.keys(), key=lambda x: abs(x - target))
            selected.append(available[closest])
        trace_files = sorted(set(selected))

    results = []
    for tf in trace_files:
        try:
            results.append(analyse_trace(tf))
        except Exception as e:
            print(f'  ERROR in {tf}: {e}')

    return results

def summarise_group(traces, label):
    if not traces:
        return f'  {label}: no data'
    n = len(traces)
    n_deal = sum(1 for t in traces if t['outcome'] == 'deal')
    avg_score = sum(t['score'] for t in traces) / n
    total_actions = sum(t['n_actions'] for t in traces)
    n_trunc = sum(t['n_truncated'] for t in traces)
    n_coll  = sum(t['n_collapsed'] for t in traces)
    avg_len_list = [l for t in traces for l in t['lengths']]
    avg_len = sum(avg_len_list) / len(avg_len_list) if avg_len_list else 0
    n_has_alloc = sum(t['n_has_alloc'] for t in traces)
    n_no_alloc  = sum(t['n_no_alloc']  for t in traces)
    na_fracs = [t['na_frac'] for t in traces]
    avg_na_frac = sum(na_fracs) / len(na_fracs)
    n_special = sum(t['n_special_tok'] for t in traces)
    n_roleplay = sum(t['n_roleplay'] for t in traces)
    fail_modes = Counter(t['failure_mode'] for t in traces if t['outcome'] == 'no_deal')

    lines = [
        f'  {label} ({n} traces):',
        f'    deal rate: {n_deal}/{n} ({100*n_deal/n:.0f}%)',
        f'    avg score: {avg_score:.2f}',
        f'    actions: total={total_actions}, truncated={n_trunc} ({100*n_trunc/total_actions:.0f}%), '
        f'collapsed={n_coll} ({100*n_coll/total_actions:.0f}%)',
        f'    avg action char length: {avg_len:.1f}',
        f'    actions with allocation: {n_has_alloc}/{total_actions} ({100*n_has_alloc/total_actions:.0f}%)',
        f'    actions WITHOUT allocation: {n_no_alloc}/{total_actions} ({100*n_no_alloc/total_actions:.0f}%)',
        f'    avg non-ASCII char fraction: {avg_na_frac:.3f}',
        f'    actions with special tokens: {n_special}',
        f'    actions with roleplay artifacts: {n_roleplay}',
        f'    no_deal failure modes: {dict(fail_modes)}',
    ]
    return '\n'.join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("NEGOTIATION TRACE ANALYSIS")
    print("=" * 80)

    # ── 1. Suite sampling: 2-3 runs per suite ──────────────────────────────
    SUITE_RUNS = {
        'negotiation_experiments': [
            '02_qwen_0.5b',
            '05_qwen_7b_lora_r16',
            '11_long_dialogue',
            '13_large_token_budget',
        ],
        'negotiation_lora_shared_experiments': [
            '01_bare_no_shuffle_r16',
            '04_marcus_sophia_shuffle_r16',
            '07_bare_no_shuffle_r64',
        ],
        'negotiation_population_experiments': [
            '01_2agent_char_lora_r16',
            '04_4agent_char_lora_r16_shuffle',
            '09_4agent_char_independent',
        ],
        'negotiation_prompt_experiments': [
            '01_same_marcus_no_shuffle_sep',
            '04_marcus_sophia_shuffle_sep',
            '09_marcus_sophia_no_shuffle_shared',
        ],
        'negotiation_role_experiments': [
            '01_no_shuffle_lr3e5_kl0',
            '05_no_shuffle_lr1e5_kl01',
            '09_shared_no_shuffle_lr1e5_kl01',
        ],
    }

    SAMPLE_ITERS = [10, 50, 200, 400, 500]  # early, mid, late

    suite_all_results = {}  # suite -> {run -> [traces]}

    for suite, runs in SUITE_RUNS.items():
        print(f"\n{'='*70}")
        print(f"SUITE: {suite}")
        print(f"{'='*70}")
        suite_all_results[suite] = {}

        for run_name in runs:
            run_dir = f'{RUNS_ROOT}/{suite}/{run_name}'
            if not os.path.isdir(run_dir):
                print(f"  SKIP (not found): {run_dir}")
                continue

            traces = analyse_run(run_dir, sample_iters=SAMPLE_ITERS)
            if not traces:
                print(f"  SKIP (no traces): {run_name}")
                continue

            suite_all_results[suite][run_name] = traces

            early = [t for t in traces if t['iteration'] <= 50]
            late  = [t for t in traces if t['iteration'] >= 400]

            print(f"\n  Run: {run_name}")
            print(f"  Iterations sampled: {[t['iteration'] for t in traces]}")
            print(summarise_group(early, 'EARLY (iter<=50)'))
            print(summarise_group(late,  'LATE  (iter>=400)'))

            # Per-trace detail
            for tr in traces:
                itr = tr['iteration']
                out = tr['outcome']
                sc  = tr['score']
                fm  = tr['failure_mode']
                na  = tr['na_frac']
                coll = tr['n_collapsed']
                na_txt = f', na={na:.3f}' if na > 0.01 else ''
                coll_txt = f', collapsed={coll}/{tr["n_actions"]}' if coll > 0 else ''
                special_txt = f', special_toks={tr["n_special_tok"]}' if tr['n_special_tok'] else ''
                print(f"    iter={itr:04d}: {out} score={sc} fm={fm}{na_txt}{coll_txt}{special_txt}")

    # ── 2. Full longitudinal scan: negotiation_experiments/05_qwen_7b_lora_r16 ──
    print(f"\n{'='*70}")
    print("LONGITUDINAL SCAN: negotiation_experiments/05_qwen_7b_lora_r16")
    print(f"{'='*70}")

    long_run_dir = f'{RUNS_ROOT}/negotiation_experiments/05_qwen_7b_lora_r16'
    long_traces = analyse_run(long_run_dir, sample_iters=None)  # all traces

    if long_traces:
        # Group into bands
        early_band = [t for t in long_traces if t['iteration'] <= 50]
        mid_band   = [t for t in long_traces if 100 <= t['iteration'] <= 250]
        late_band  = [t for t in long_traces if t['iteration'] >= 400]

        print(summarise_group(early_band, 'EARLY band (iter 10-50)'))
        print(summarise_group(mid_band,   'MID   band (iter 100-250)'))
        print(summarise_group(late_band,  'LATE  band (iter 400-500)'))

        # Deal rate over time
        print("\n  Per-iteration outcomes:")
        for t in sorted(long_traces, key=lambda x: x['iteration']):
            itr   = t['iteration']
            out   = t['outcome']
            sc    = t['score']
            fm    = t['failure_mode']
            na    = t['na_frac']
            coll  = t['n_collapsed']
            na_txt = f' na={na:.3f}' if na > 0.01 else ''
            coll_txt = f' collapsed={coll}/{t["n_actions"]}' if coll > 0 else ''
            alloc_txt = f' alloc={t["n_has_alloc"]}/{t["n_actions"]}' if t["n_has_alloc"] > 0 else ''
            sp_txt = f' spec={t["n_special_tok"]}' if t['n_special_tok'] else ''
            print(f"    iter={itr:04d}: {out:7s} sc={sc:2d} fm={fm:25s}{na_txt}{coll_txt}{alloc_txt}{sp_txt}")

    # ── 3. Detailed failure mode examples ──────────────────────────────────
    print(f"\n{'='*70}")
    print("DETAILED FAILURE MODE EXAMPLES")
    print(f"{'='*70}")

    # Collect interesting examples across all scanned traces
    all_traces_flat = []
    for suite_data in suite_all_results.values():
        for run_traces in suite_data.values():
            all_traces_flat.extend(run_traces)
    if long_traces:
        all_traces_flat.extend(long_traces)

    # A) Collapsed / whitespace actions
    print("\n[A] COLLAPSED ACTIONS (pure whitespace / newline spam)")
    collapsed_examples = [(t, a) for t in all_traces_flat
                          for a in t['example_actions'] if is_collapsed(a)]
    if collapsed_examples:
        tr, ex = collapsed_examples[0]
        print(f"  Example from iter={tr['iteration']}: {repr(ex[:80])} ...")
        print(f"  (all {tr['n_collapsed']} out of {tr['n_actions']} actions in that trace collapsed)")
    else:
        print("  None found in sample.")

    # B) Truncated actions (end mid-word)
    print("\n[B] TRUNCATED ACTIONS (end mid-word/mid-number)")
    trunc_examples = []
    for t in all_traces_flat:
        for a in t['example_actions']:
            if is_truncated(a):
                trunc_examples.append((t, a))
    if trunc_examples:
        for tr, ex in trunc_examples[:3]:
            print(f"  iter={tr['iteration']}: {repr(ex[-100:])}")
    else:
        print("  None found in sample.")

    # C) Language drift / non-ASCII
    print("\n[C] LANGUAGE DRIFT (high non-ASCII)")
    na_examples = sorted(all_traces_flat, key=lambda t: t['na_frac'], reverse=True)
    for t in na_examples[:5]:
        if t['na_frac'] > 0.01:
            # Find the worst individual action
            worst_action = max(
                (a for a in t['example_actions'] if a),
                key=non_ascii_fraction,
                default=''
            )
            # Also check submission texts
            for st in t['submission_texts']:
                if non_ascii_fraction(st) > non_ascii_fraction(worst_action):
                    worst_action = st
            print(f"  iter={t['iteration']} na_frac={t['na_frac']:.3f} scripts={list(t['scripts'])[:8]}")
            print(f"    example: {repr(worst_action[:250])}")

    # D) Special tokens bleeding through
    print("\n[D] SPECIAL TOKENS BLEEDING THROUGH")
    sp_examples = [(t, ex) for t in all_traces_flat
                   for ex in t.get('special_examples', [])]
    seen = set()
    for t, ex in sp_examples[:5]:
        key = repr(ex[:80])
        if key not in seen:
            seen.add(key)
            toks = SPECIAL_TOKEN_RE.findall(ex)
            print(f"  iter={t['iteration']} tokens={toks}: {repr(ex[:200])}")

    # E) Allocations in submissions
    print("\n[E] SUBMISSION ALLOCATION EXAMPLES")
    alloc_traces = [t for t in all_traces_flat if any(t['sub_has_alloc'])]
    no_alloc_both = [t for t in all_traces_flat
                     if t['outcome'] == 'no_deal' and not any(t['sub_has_alloc'])]
    print(f"  Traces where >=1 submission has allocation: {len(alloc_traces)}")
    print(f"  no_deal traces where NEITHER submission has allocation: {len(no_alloc_both)}")
    for t in alloc_traces[:3]:
        for i, (st, ha) in enumerate(zip(t['submission_texts'], t['sub_has_alloc'])):
            allocs = extract_allocations(st)
            print(f"  iter={t['iteration']} sub[{i}] has_alloc={ha} allocs={allocs}: {repr(st[:150])}")

    # F) Roleplay artifacts
    print("\n[F] ROLEPLAY ARTIFACTS")
    rp_examples = [(t, a) for t in all_traces_flat
                   for a in t['example_actions'] if has_degenerate_roleplay(a)]
    if rp_examples:
        for tr, ex in rp_examples[:3]:
            print(f"  iter={tr['iteration']}: {repr(ex[:200])}")
    else:
        print("  None found in sample.")

    # ── 4. Failure mode breakdown ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAILURE MODE BREAKDOWN (all sampled traces)")
    print(f"{'='*70}")
    fm_counter = Counter(t['failure_mode'] for t in all_traces_flat)
    outcome_counter = Counter(t['outcome'] for t in all_traces_flat)
    print(f"  Outcomes: {dict(outcome_counter)}")
    print(f"  Failure modes (no_deal traces): {dict(fm_counter)}")

    # ── 5. Script/language breakdown ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("LANGUAGE/SCRIPT BREAKDOWN")
    print(f"{'='*70}")
    all_scripts = Counter()
    for t in all_traces_flat:
        for s in t['scripts']:
            all_scripts[s] += 1
    print("  Top script blocks appearing in action text:")
    for script, cnt in all_scripts.most_common(20):
        print(f"    {script}: {cnt} traces")

    # ── 6. Cross-suite comparison ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-SUITE SUMMARY")
    print(f"{'='*70}")
    for suite, suite_data in suite_all_results.items():
        all_suite_traces = [t for run_traces in suite_data.values() for t in run_traces]
        if not all_suite_traces:
            continue
        n = len(all_suite_traces)
        n_deal = sum(1 for t in all_suite_traces if t['outcome'] == 'deal')
        total_actions = sum(t['n_actions'] for t in all_suite_traces)
        n_coll = sum(t['n_collapsed'] for t in all_suite_traces)
        n_has_alloc = sum(t['n_has_alloc'] for t in all_suite_traces)
        avg_na = sum(t['na_frac'] for t in all_suite_traces) / n
        n_special = sum(t['n_special_tok'] for t in all_suite_traces)
        print(f"\n  {suite}:")
        print(f"    deal_rate={100*n_deal/n:.0f}%  collapsed={100*n_coll/total_actions:.0f}%  "
              f"alloc_in_action={100*n_has_alloc/total_actions:.0f}%  "
              f"avg_na={avg_na:.3f}  special_tok_traces={n_special}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
