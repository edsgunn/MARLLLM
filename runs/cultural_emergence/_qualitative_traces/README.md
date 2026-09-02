# Annotated population scatters & organised iter-200 traces

This bundle exists so you can take a point on `fig1_scatter_kl_vs_percloss_iter200`
(or its sister figures), see *which* agent it represents, and pull up that
agent's actual posts in seconds. The original (un-annotated) figures in
`_population_figures/` are untouched.

## Where things live

```
_population_figures_annotated/        # labelled scatters (this is the entry point)
  fig1_annotated_smallmultiples_kl_vs_percloss.{pdf,png}
  fig1_annotated_overlay_kl_vs_percloss.{pdf,png}
  fig3_annotated_smallmultiples_kl_vs_entropy.{pdf,png}
  fig4_annotated_smallmultiples_entropy_vs_percloss.{pdf,png}
  agent_index.csv

_qualitative_traces/                  # per-agent post extracts
  posts_by_run/<NNagent_substrate>/<First_Last>.md
  posts_by_regime/<regime>/<run>__<First_Last>.md     (symlinks)
  raw_traces/<NNagent_substrate>_traces.txt           (whole-run dump)
  extracted_index.csv
```

Run tags are uniform: `02agent_ashbourne`, `04agent_strathearn`, etc. Substrate
abbreviations match the original labels:

- `ashbourne` = `study_group_ashbourne_gc`
- `strathearn` = `study_group_strathearn_server`

## How to use the annotated figure

The most useful view is the **small-multiples** PDF for KL × perception loss
(`fig1_annotated_smallmultiples_kl_vs_percloss.pdf`). Eight panels, 4 population
sizes × 2 substrates. Every dot is labelled with the agent's first name.
Background shading still encodes regime (green = coherent, blue = committed,
sand = mid-range, pink = degenerate).

If a name appears in multiple panels (e.g. `Priya` shows up in 2/4/8/16-agent
ashbourne), each instance is a *separate* training run with the same character
prompt — they share only the role/biography text. Treat them as distinct
agents.

The **overlay** version is the original single-panel layout but with every
point tagged. Use this when you want the cross-population perspective in one
view; it's crowded but readable. Tags are formatted `Firstname NA` (N =
population size, A = first letter of substrate, e.g. `Mhairi 16S`).

The star outline (e.g. on Priya in 16-agent ashbourne) marks
trace-confirmed-coherent agents — independent verification that the numerical
classification matches the trace text.

## How to look up an agent

1. Click on the dot in the annotated PDF (or read its label).
2. Open `posts_by_run/<run-tag>/<First_Last>.md`. Each file has YAML-like
   metadata at the top (regime, KL, perception loss, entropy, iter) and the
   verbatim post text grouped by episode.
3. For diagnostic-grade inspection (raw assistant generations, including
   unclosed `</post>` token loops that don't show up in the clean post text),
   open `raw_traces/<run-tag>_traces.txt` and search for `AGENT <full name>`.
   That section preserves the model's full output, system prompts, and
   any token-level pathology.

The `posts_by_regime/` tree contains the same per-agent files cross-referenced
by their classification. Useful for sweeps like "show me every committed agent
in one go."

## Spot-checks already run

`trace_verification_iter200.md` (in `_population_figures/`) and
`_absorption_by_regime/QUALITATIVE_TRACE_VALIDATION.md` between them have
worked through one example per regime against this dataset. Headline:

| Regime | What the trace shows |
|---|---|
| `coherent_low_kl` | fluent, on-task, lightly character-flavoured |
| `committed` | terse compressed persona; sometimes drifting into a stylised idiolect |
| `mid_range` | partial collapse — agents emit short stock phrases (`"Vacuous? Yuki?"`) that are topical but content-free |
| `degenerate` (entropy-driven) | post text often still readable; underlying generation is collapsed (`</post>\n</python>` loops past the post body) |
| `degenerate` (perception-driven) | post text broken: literal token loops, or fluent-but-empty boilerplate that has erased character distinctions |

The two failure modes inside the `degenerate` bucket are not the same thing —
worth keeping in mind when pulling examples for the paper. The
`fig4_annotated_smallmultiples_entropy_vs_percloss.pdf` is the cleanest view of
the split: low-entropy + low-perception is *policy collapse*, high-perception
is *text degeneration*, and the corner with both is rare.

## Reproducing

```
.venv/bin/python scripts/make_annotated_population_figures.py
.venv/bin/python scripts/extract_iter200_traces.py
```

Requires `classification_table_iter200_*.csv` in `_population_figures/`
(written by `plot_population_regime_figures.py`).

## Caveats

- 16-agent panels in the small-multiples view are dense; some labels overlap
  near the cluster at `(KL≈0.5, perc≈1.5)`. The annotated overlay PDF spreads
  these out a bit better, and the per-cell `n=` count in the corner of each
  panel lets you sanity-check that every agent is plotted.
- A handful of agents have mid-iter and end-iter regime mismatches (33/88 in
  the absorption analysis). For the iter-200 figures specifically this is
  less of a concern because the absorption window doesn't apply, but for any
  cross-iter analysis (e.g. lining iter-25 traces up with iter-200 metrics) be
  aware that an agent's class can have changed in between.
- The `raw_traces/*.txt` files are large (5–25k lines) and contain the
  original chat-template formatting. They're committed verbatim; if you want
  per-agent slices, parse on the `AGENT <full name>` heading lines in each
  file.
