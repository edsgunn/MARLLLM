# B6 — what does `thinking_frac` actually contain?

|  |  |
|---|---|
| Script | [`scripts/tier_b/b6_token_accounting_audit.py`](../../../scripts/tier_b/b6_token_accounting_audit.py) |
| Inputs | checkpoint traces + `metrics.jsonl`, 5 long forum runs, early and late |
| Ran | 2026-09-01, revised the same day after a correction |

## Correction to the first version of this note

The first version claimed `thinking_frac` "does not measure private
deliberation" because there are no `<think>` tags in any forum run. **That was
wrong, and the reasoning was backwards.** In this framework `<think>` is not what
makes a token private. The environment routes by tag: `<post>` is broadcast,
`<python>` is executed and its output broadcast, and "everything else — private
thinking; never shown to" other agents
([`envs/forum/env.py`](../../../envs/forum/env.py) lines 290, 518, 689).
`<think>` exists only so that a `<post>` nested inside it does not count
(env.py:677). So `thinking_tokens = max(0, n_act − post_tokens)` is the correct
implementation of the environment's own definition of private, and the absence
of `<think>` tags says nothing at all about whether the private channel is used.

What is actually wrong with the metric is different, and there are two things.

## Headline

**1. On tool substrates, most early "thinking" is the *public* python channel.**
`<python>` blocks sit outside `<post>`, so they count as thinking — but pybot
broadcasts their output to every agent. Share of non-post tokens that is
`<python>`, at the second checkpoint:

| substrate | early | late |
|---|---:|---:|
| margin_notes (tool) | **80%** | 2% |
| inner_loop (tool) | **68%** | 7% |
| ashbourne_gc 16a | 38% | 6% |
| ashbourne_gc 32a | 33% | 2% |
| strathearn 16a | 11% | 0% |

So a `thinking_frac` of 0.41 on margin_notes at iteration 100 is mostly tool use,
not deliberation. The metric conflates the most public channel in the substrate
with the private one.

**2. Late, a large genuine pre-post region appears — and a substantial part of
it is posts the model wrote and failed to tag.** Mean tokens before the first
`<post>`, early → late: margin_notes 0 → 162, inner_loop 2 → 232, strathearn
1 → 67, ashbourne_gc 0 → 35. Only 0–11% of those regions are degenerate tag or
token loops. The rest is prose, and reading a fair sample of it
([`pre_post_excerpts.txt`](pre_post_excerpts.txt)) shows three kinds:

- **Untagged posts.** Kai Dempsey, inner_loop iteration 325, with no `<post>`
  anywhere in the turn: *"Type differences matter deeply. Let's look at the types
  involved… `a` is a list, which is mutable and has the `pop` method, while `b`
  is a tuple, which is immutable…"* — addressed to the forum, on topic, in
  persona. Roisin Doherty on Strathearn writes *"Claim: … Reasoning: … Who:
  mira"*, which is that substrate's **pinned posting format**. These agents wrote
  posts. The environment discarded them.
- **Genuine deliberation.** *"Roisin noticed the thread was getting increasingly
  repetitive… she crafted a concise reply focusing on orderly problem-solving"* —
  third-person narration about posting, not a post.
- **Degeneration.** Token loops (`</?</?</?…`), 8–11% of late generations on the
  two tool substrates and ~0% elsewhere.

**Tags and intent have come apart in both directions.** The converse case is on
record too: at margin_notes iteration 300, *"Note to self: Kai likely means to
question something briefly…"* was emitted **inside** `<post>` and published to
the whole forum.

## What this settles

- **`thinking_frac` measures the environment's private channel correctly, but
  its name is wrong twice over**: it includes `<python>`, which is public, and it
  cannot distinguish deliberation from an untagged post. — *established*
- **`post_missing` turns include turns where the agent wrote a well-formed post
  and did not tag it.** — *established* by direct reading
- **Structural tag compliance decays over training.** `<post>` opens per close go
  1.7 → 87 (32-agent), 1.8 → 28.8 (16-agent margin notes); one run inverts to
  0.32 (three closes per open, the `</post>` loop). — *established*

## What it does not settle

**How the three kinds divide.** The sample is read, not counted; a classifier or
a hand-coded sample would be needed to put a fraction on "untagged post" versus
"deliberation". That is the number that decides whether the closing public
channel (B4) is a policy pathology or a substrate one.

Nor does it explain *why* compliance decays. One candidate worth stating:
**nothing in the objective rewards a well-formed `<post>`.** The perception loss
is on observation tokens and the action loss is REINFORCE on future observation
surprise; a closing tag earns neither. Format compliance is held up only by the
KL anchor to the base model, and it decays as the policy drifts away from it.

## Status

`CLOSED` as an audit. The consequence for [B4](../b4_public_channel/NOTES.md) is
recorded there: content not reaching other agents is measured correctly, but the
cause is mixed and partly a tagging failure.

---
*Analysed 2026-09-01. Machine output stays local.*
