# Sanctioning Measurement Framework (Rungs 0–1)

Reduces *"do cultural norms emerge?"* to a quantitative question by measuring
**sanctioning** — approving/disapproving reactions between agents — in
forum-style multi-agent training transcripts. A norm (per Leibo's theory of
appropriateness) is a stable mapping from a behaviour-in-context to a consistent
sanctioning response by others.

- **Rung 0** — *Does sanctioning occur at all?* Base rate of non-neutral
  reactions per iteration (approval / disapproval split out).
- **Rung 1** — *Is sanctioning targeted (vs. mood)?* Does the same behaviour
  type reliably draw the same valence?

Rungs 2–4 (cross-sanctioner consistency, second-order sanctioning, evolution)
are out of scope, but the schema retains `target_agent_id`, `agent_id`, and
`iteration` so they are pure aggregation additions later.

## Design commitments (load-bearing)

1. **Detection ≠ interpretation.** The LLM judge does per-event labelling only
   (valence, target, free-text trigger description). All "is this a norm" logic
   lives in deterministic aggregation (`rung0.py`, `rung1.py`).
2. **No behaviour taxonomy in the prompt.** The judge emits free-text trigger
   descriptions; behaviour categories are formed *post hoc* by clustering.
3. **Non-prescriptive prompting** (`prompts/sanction_judge_v1.txt`): valence is
   defined operationally, not morally; sanctioning is never implied to be
   expected. Low rates are a finding, not a bug.
4. **The judge is an instrument and must be validated** before Rung-1 numbers
   are trusted. `rung1` is gated on a passing `validate` report for the same
   `(provider, model, prompt_version)`.
5. **Provider-agnostic** — Anthropic and OpenAI behind one interface.

## Layout

| File | Role |
|---|---|
| `adapter.py` | **Only** file that knows the on-disk wire format → `Episode`/`Post` |
| `prompts/sanction_judge_v1.txt` | Versioned, non-prescriptive judge prompt |
| `backends.py` | Anthropic/OpenAI judge backends + OpenAI embeddings; pricing |
| `cache.py` | SQLite cache (judge calls + embeddings); re-runs don't re-bill |
| `judge.py` | Prompt build, call, strict-JSON parse + validate, one retry |
| `persistence.py` | Append-only JSONL event records; separate cluster-assignment store |
| `extract.py` | `extract` command (threaded, resumable, cost tally) |
| `rung0.py` | Rung-0 table + plots |
| `cluster.py` | Embed + cluster trigger descriptions (tfidf/openai; agglomerative/hdbscan) |
| `rung1.py` | Rung-1 consistency + MI (permutation null) + threshold sweep; gated |
| `validate.py` | `label` (human helper) + `validate` (scoring + gate report) |
| `cli.py` | `python -m sanctioning <cmd>` |

Config: [`configs/sanctioning.yaml`](../configs/sanctioning.yaml). The corpus is
the two confirmed 5-seed forum families — `ashbourne_gc`
(`run8_16agent_7B_study_group_ashbourne_gc` + seed2–5) and `margin_notes`
(`run7_8agent_7B_margin_notes` + seed2–5).

## Running it

Use the hand-built venv (`.venv/bin/python`; `uv run` resyncs and breaks it).
Set the key first:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # or OPENAI_API_KEY for the openai backend
cd /lus/lfs1aip2/projects/a5l/egunn/projects/MARLLLM
PY=".venv/bin/python"; CFG="configs/sanctioning.yaml"

# 1. Extract sanctioning events over the whole corpus (~6.4k posts; cached/resumable).
#    Smoke test first if you like:  $PY -m sanctioning extract --config $CFG --limit 20
$PY -m sanctioning extract --config $CFG

# 2. Rung 0 — base rates (no gate). Tables + plots in runs/sanctioning/.
$PY -m sanctioning rung0 --config $CFG

# 3. Cluster trigger descriptions (post hoc behaviour types).
$PY -m sanctioning cluster --config $CFG

# 4. Validate the judge (REQUIRED before trusting Rung 1):
$PY -m sanctioning label    --config $CFG     # hand-label a stratified sample
$PY -m sanctioning validate --config $CFG     # score + write the gate report

# 5. Rung 1 — targeting consistency + MI. Refuses unless the gate passes.
$PY -m sanctioning rung1 --config $CFG
#   (--force emits provisional Rung-1 results without a passing gate)
```

Filters apply to every command: `--substrate ashbourne_gc`, `--run-id
margin_notes_s2` (repeatable), `--iter-min 25 --iter-max 150`.

### Outputs (under `runs/sanctioning/`)

- `events.jsonl` — one record per post, full provenance (provider/model/prompt
  version/usage). The auditable source of truth.
- `clusters.jsonl` — `event_id → behaviour_cluster_id + medoid label`.
- `rung0_table.csv`, `figures/rung0_*.png`.
- `rung1_consistency_pooled.csv`, `rung1_consistency_per_run.csv`,
  `rung1_mi_by_iteration.csv`, `rung1_mi_threshold_sweep.csv`,
  `figures/rung1_mi_vs_iteration.png`.
- `validation/report_<provider>__<model>__<prompt_version>.json` — the gate.
- `cache.sqlite` — judge + embedding cache.

## Notes / caveats baked into the outputs

- **Disapproval ≈ 0 is a result.** Rung 0 reports approval and disapproval
  separately; a flat-near-zero disapproval rate across iterations is the
  chatbot-base finding, not a pipeline failure.
- **Clustering is a researcher degree of freedom.** `rung1` emits cluster count,
  min cluster size, and an MI sensitivity sweep over the threshold so a reviewer
  can see the result is not a granularity artefact.
- **Embeddings:** default `tfidf` needs no API key (a lexical fallback). For
  semantic clustering set `embedding.provider: openai` (needs `OPENAI_API_KEY`);
  vectors are cached per text.
- **Validate both providers** (run `validate` under two configs) to quantify
  judge-choice sensitivity, per the spec.
- **Deviation from the spec's literal output schema:** the judge returns
  `{"events": [...]}`; the reacting-post key is attached deterministically from
  known provenance rather than asked of the model (it is not a judgement). All
  other schema fields are exactly as specified.
