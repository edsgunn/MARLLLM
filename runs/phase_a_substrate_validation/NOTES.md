# phase_a_substrate_validation

*The framework's first pre-registered experiment. **The family is the unit of
inference, and the family is half-run.***

|  |  |
|---|---|
| Configs | [`configs/phase_a_substrate_validation/`](../../configs/phase_a_substrate_validation/) — 13 configs, and its `README.md` states the claim and the matrix |
| Substrate | `deal_or_no_deal`, two agents |
| Model | `Qwen2.5-1.5B-Instruct` (with one exception, below) |
| Ran | conditions 1, 2 and 3 — 8 of 13 configs |
| Never ran | conditions 4 (action-only), 5 (symmetric CCSM), 6 (7B LoRA) |
| Never evaluated | every trained checkpoint |

## What it was for

The Phase A spec states one claim, and the config directory's `README.md` states
it in full:

> Character-Conditioned Surprise Minimisation, applied to a pretrained language
> model in a multi-agent text environment, produces non-trivial agentic
> behaviour that (a) differs from the pretrained baseline, (b) varies with the
> character prompt, and (c) improves on environment-relevant external metrics
> over training.

Each condition was to be run twice, once under a cooperative and once under a
competitive character prompt; that prompt swap is the diagnostic the claim turns
on. Two helper scripts were to score every trained checkpoint —
`evaluate_negotiation.py` for (c) and `analyze_prompt_conditional.py` for (a)
and (b).

## Headline

**The training ran and the experiment did not.** Six training runs completed
cleanly, to 500 iterations, with a real and repeatable effect — perception loss
falls by roughly half in every arm. But:

- **No trained checkpoint was ever scored.** There is no `eval_final.json` and no
  `prompt_conditional_post.json` anywhere under `runs/`. Criteria (a) and (b)
  are untested.
- **Criterion (c) was close to unmeasurable before anything was trained.** The
  pretrained baseline reaches agreement on **3.1%** (competitive) and **4.7%**
  (cooperative) of 64 episodes. An external metric that fires two or three times
  in sixty-four cannot show an improvement.
- **The prompt swap at seed 42 is confounded.** `02_ccsm_focal_competitive` ran
  on `Qwen/Qwen2.5-1.5B` — the *base* model — while its cooperative partner ran
  Instruct. The tracked config was later edited to say Instruct (commit
  `92e5d76`), so the config in git no longer describes the run.
- **The ablation that would have interpreted the result was never run.**
  Conditions 4 and 5 have configs and no runs.

What the family *does* establish it establishes well, and it is not what it set
out to establish.

## What this settles

- **CCSM reduces perception loss substantially and repeatably.** ~1.1–1.4 →
  0.38–0.64 across four condition-2 arms and two condition-3 arms, over 500
  iterations. — *established*
- **The perception term is sufficient for that.** Perception-only reaches the
  same perception loss with KL from base of 0.07 against 0.26–0.70, and with
  entropy intact. — *established*
- **The action term is what collapses the policy.** Both seed-42 full-loss arms
  end with entropy ≤ 0.003; both perception-only arms end at 1.7–2.0. —
  *likely*
- **The headline condition is seed-unstable.** Identical configuration, seeds 42
  and 137, opposite sides of the `policy_collapse` boundary. — *established*
- **`deal_or_no_deal` at 1.5B is a floor-effect substrate for agreement rate.**
  — *established*

## What it does not settle

Its own claim, in all three parts. What is needed is not more training: it is
`evaluate_negotiation.py` and `analyze_prompt_conditional.py` run over the
checkpoints that already exist, and a criterion with headroom.

## Status

`CLOSED` as a training campaign. **`OPEN` as an experiment** — the checkpoints
are on disk and the evaluation is cheap. See the open questions in `STATE.md`.

---
*Analysed retrospectively on 2026-09-01 from `metrics.jsonl`, `config.json` and
`eval.json`. Machine output stays local.*
