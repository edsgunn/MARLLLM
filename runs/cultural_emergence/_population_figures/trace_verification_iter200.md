# Trace verification of iter-200 classifications

Cross-check of the `(KL, perception loss, entropy)`-based classification used in
`fig1`–`fig5` against the actual checkpoint traces (`traces/ckpt_000200/traces.txt`)
for each run.

**Headline.** The numerics agree with behaviour, with one important caveat: the
`degenerate` bucket conflates two phenomenologically distinct failure modes —
*text-level* degeneration (gibberish output) and *policy-level* collapse (low
entropy, surface-coherent text).

## Cases that clearly match

- **8-agent ashbourne, Priya Shah** — `degenerate`, perc=2.68, ent=0.03.
  Outputs literal token-loop garbage:

  ```
  '?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?</?...'
  '?");\n\nutow?'
  ',Th?</post?</?</?</?</?</?</?</?...'
  ```

  Classification unambiguous.

- **4-agent ashbourne, Priya Shah** — `degenerate`, perc=1.19, ent=0.08.
  Chinese clause-loops followed by long runs of blank lines:

  ```
  '考虑了一些例子，确实 \(n=1\) 是一个特例。\n\n特意检查了 \(n=1\) 的情况。\n\n
   确认了几组数对。\n\n重新整理思路。\n\n确保n=1的情况明确。\n\n…\n\n\n\n\n\n\n\n…'
  '整理思路，明确结构。\n\n逐步确认。\n\n简化表述。\n\n直接总结。\n\n明确结论。'
  ```

  Collapsed.

- **16-agent ashbourne, Priya Shah** at iter 200 — `coherent_low_kl`,
  the trace-confirmed star. Fully coherent, on-character, on-task:

  > Rhys is on the right track but forgot about 0 being a quadratic residue.
  > Also, his pairing argument needs a bit more clarity. We need to prove there
  > are exactly \\(\\frac{p+1}{2}\\) quadratic residues modulo \\( p \\). Rhys
  > mentioned pairs \\( (a, -a) \\). Correct, but we need to include 0.

  Star is well-placed.

- **16-agent ashbourne, committed agents** (Marcus Webb kl=4.96, Jonah
  Liefferink, Imogen Carter): genuinely engaged math, but in a characteristic
  compressed style:

  > agree. brute-forcing divisors works. try p=11, a=2: 1,2,5,10→n=10. check
  > others too. brute-force=proof precursor?
  >
  > i follow. so \\(n\\) divides \\(p-1\\)? but how do we know which div?
  > — Marcus

  This is exactly the "committed = high-KL drift but coherent perception"
  prediction the paper makes.

- **8-agent ashbourne, mid-range agents** (Marcus Webb perc=1.74, Hana, Imogen,
  Sam): visibly drifting, partially broken, partially fluent. E.g. Marcus Webb:

  ```
  'sorry q&amp;pping, but...'
  '1.3+2=5\n2.9-4=5\n3.11+10=9\n4.2-5=9\n5.7+12=7'
  ```

  Hana / Imogen partway through have pivoted into Chinese for a Glasgow-school
  substrate. Mid-range is the right call.

## The caveat: low-entropy `degenerate` ≠ high-perception `degenerate`

Five agents are flagged `degenerate` purely by `entropy < 0.2`, while their
perception loss stays low:

| run                    | agent             | KL    | perc  | ent   | trace behaviour |
|------------------------|-------------------|-------|-------|-------|------------------|
| 2-agent strathearn     | Mhairi Buchanan   | 0.11  | 0.84  | 0.07  | post text reads coherent; post-`</post>`, model loops `</post>\n</python>` until cutoff |
| 2-agent strathearn     | Callum Reid       | 0.13  | 1.59  | 0.13  | same loop pattern |
| 4-agent strathearn     | Niamh Donnelly    | 0.15  | 0.85  | 0.18  | post text fluent ("Thanks for the thorough verification, everyone!") |
| 16-agent strathearn    | Daniyal Khan      | 0.27  | 1.44  | 0.12  | terse fragmentary mix of English and Chinese math notes |
| 16-agent strathearn    | Sumi Tanaka       | 0.16  | 1.53  | 0.16  | fluent technical posts on binomial expectations |

Example of Mhairi's underlying generation (iter 200, 2-agent strathearn) — the
posted text is fine, but the assistant span loops the closing tags forever:

```
<post>
Check if there are any other simple cases to consider, like \( n = -40 \) or
\( n = -42 \). Also, think about the general approach: can we find a pattern
or a theorem that might help?
</post>
</post>
</python>
…
</post>
</python>
</post>
</python>
</post>
</python>
…
```

The numerical call (entropy collapse → `degenerate`) is correct *at the policy
distribution level* — the distribution has narrowed to a near-deterministic
mode. But to a human reader the forum text these agents emit is largely
acceptable. The same `degenerate` bucket also contains 8-agent-ashbourne Priya
outputting `?</?</?...` literal garbage. **Those two failure modes are not the
same thing in the paper's narrative.**

## Borderline cases worth flagging

- **Felix Chen** (16-agent ashbourne, `committed`, kl=6.05): posts include
  `偶數,SIGNAL` and `smallest \(n\)查---`. Numerical thresholds keep him in
  `committed` but visible token corruption is starting to creep in.

- **2-agent ashbourne Tom Whitaker** (`committed`): writes fluent Chinese
  math-talk in an English Glasgow sixth-form server:

  ```
  'n=1确实是特例，x ≡ a (mod 1) 对任意 x 都成立。我们先讨论 n > 1 的情况，
   而 n = 1 时 x = a 显然成立，但不提供额外信息。…'
  ```

  High KL is real (full substrate language-switch); perception loss low (the new
  mode compresses well). Per the paper's definition this is exactly committed,
  but it's *language-switch commitment* rather than *style commitment* — a
  paper-level distinction worth deciding on explicitly.

## Suggested adjustment to the classification

Either:

1. **Split `degenerate` into two categories** in the script:

   - `text_degenerate`: `perc_loss >= 2.0`
   - `policy_collapse`: `entropy < 0.2` (or `> 1.8`) with `perc_loss < 2.0`

   This matches behaviour more honestly, and Fig 2 would then show that 2-agent
   runs *collapse the policy without ruining the text* — which is a cleaner
   population-size story than the current "2-agent strathearn = 100% degenerate"
   bar implies.

2. Or **keep one `degenerate` bucket** but state in the caption that it lumps
   two failure modes together, and rely on Fig 4 (perc vs entropy) for the
   reader to disambiguate.

Option 1 is recommended; the current single-bucket framing risks under-selling
the population-size finding (small populations preserve the perception
distribution but lock the policy).
