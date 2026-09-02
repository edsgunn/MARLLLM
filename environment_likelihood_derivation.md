# Recovering the environment likelihood in a multi-agent population

A companion to [`surprise_minimisation_derivation.md`](surprise_minimisation_derivation.md),
whose notation it uses. That document drops the term
$\sum_s \sigma_s \ln p_{\text{env}}(x_s \mid x_{<s})$ on the grounds that
"$p_{\text{env}}$ is only ever sampled, not evaluated". In a population of
language-model agents that is false for the only observation tokens carrying any
entropy — the other agents' utterances — and this document works out what
recovering them costs.

The question it answers: **an observation token was generated under a context the
observer cannot see, so what adjustment does the generator's own log-probability
need before it can serve as $p_{\text{env}}$ for a given observer?**

The short answer, derived below: **the adjustment is generator-specific, not
observer-specific.** Under full observability of the public thread every observer
shares the same $p_{\text{env}}$, and the single correction needed is to
marginalise the generator's private prefix. That correction is exactly a mutual
information, it is provably non-negative, and it is zero when the generator emits
no private tokens.

---

## 1. Setup

Following §2 of the main derivation: a trajectory is $x_{1:T}$ with binary types
$\sigma_t \in \{0,1\}$, $\sigma_t = 1$ for observation tokens. Each agent $i$ has
parameters $\theta_i$ (a LoRA adapter over a shared frozen base) and a character
prefix $c_i$.

The complication the main derivation elides is that in a population there is no
single trajectory. Each agent has its **own** token sequence, because each sees
its own persona, its own private deliberation, and the shared thread rendered
from its own point of view. Write:

- $u_s$ — the **public thread** prefix at position $s$: every delivered post, in
  order, with its speaker attributed. Common to all agents under full
  observability.
- $z_s$ — the generator's **private prefix** for the token about to be emitted:
  everything it generated this turn that no one else receives.
- $x^i_{<s}$ — observer $i$'s own context: a function of $u_s$, $c_i$, and $i$'s
  own private deliberation.

The generative process for an observation span is

$$j \sim \text{turn order}, \qquad z_s \sim p_{\theta_j}(\cdot \mid u_s, c_j), \qquad y_s \sim p_{\theta_j}(\cdot \mid u_s, c_j, z_s)$$

where $y_s$ is the delivered post. The environment then **discards $z_s$** and
broadcasts $y_s$ (`envs/forum/env.py`: "everything else — private thinking; never
shown to" other agents).

## 2. What is latent, and what is not

Four things differ between generator and observer. Only one of them is a latent
variable.

| | latent? | why |
|---|---|---|
| **speaker identity $j$** | **no** | posts are attributed by name in the thread |
| **generator's persona $c_j$** | **no** | deterministic given $j$; fixed for the run and known to the experimenter |
| **generator's private prefix $z_s$** | **yes** | sampled, then discarded by the environment |
| **observer's own private state** | **irrelevant** | see §3 |

So the environment's conditional, from the observer's side, is a marginal over
one variable:

$$p_{\text{env}}(y_s \mid u_s, j) \;=\; \sum_{z} P(z \mid u_s, c_j)\; p_{\theta_j}(y_s \mid u_s, c_j, z) \tag{1}$$

The remaining observation tokens are deterministic given what is visible —
structural markers, the persona prefix, and tool output given the code that
produced it — so for them $\ln p_{\text{env}} = 0$ **exactly**. After (1),
nothing is left dropped.

*(Caveat: tool output is deterministic only if the executed code is. Code using
randomness or wall-clock time is not, and is rare enough in these substrates to
handle by exception.)*

## 3. The environment conditional is observer-independent

**Claim.** For any two observers $i \ne i'$ who both see the full public thread,

$$p_{\text{env}}(y_s \mid x^i_{<s}) = p_{\text{env}}(y_s \mid x^{i'}_{<s}) = p_{\text{env}}(y_s \mid u_s, j)$$

**Proof.** $y_s$ was sampled from $p_{\theta_j}(\cdot \mid u_s, c_j, z_s)$. The
parts of $x^i_{<s}$ not contained in $u_s$ are $c_i$ and $i$'s own private
deliberation. Neither entered $j$'s conditioning set, and $j$'s sampling
randomness is drawn independently of $i$'s. Hence $y_s \perp (c_i, \text{private}_i) \mid u_s, j$,
and conditioning on them leaves (1) unchanged. $\square$

This is the direct answer to the question. **No observer-specific adjustment is
required.** The correction is a property of the generator's turn, so it is
computed **once per post** and reused by all $N-1$ observers — which is also what
makes the measurement affordable (§7).

**The exception is partial observability.** In `social_terminal` agents see only
their current room, so $u^i_s$ genuinely differs between observers and the
conditional independence above fails. There the latent set expands to include
everything that happened out of sight, and $p_{\text{env}}$ *is*
observer-specific. Nothing in this document transfers to that substrate without
rework.

## 4. The correction is a pointwise information

Let $\hat q_s = p_{\theta_j}(y_s \mid u_s, c_j, z^\star_s)$ be the computable
quantity — the generator's own likelihood under the private prefix it actually
had. By Bayes' rule on the joint over $(z, y)$ given $(u_s, c_j)$,

$$P(z \mid u_s, y_s) = \frac{P(z\mid u_s)\, p_{\theta_j}(y_s \mid u_s, z)}{p_{\text{env}}(y_s\mid u_s)}$$

Rearranging at $z = z^\star_s$ gives an **exact pointwise identity**:

$$\boxed{\;\ln p_{\text{env}}(y_s \mid u_s) \;=\; \ln \hat q_s \;-\; \underbrace{\ln \frac{P(z^\star_s \mid u_s,\, y_s)}{P(z^\star_s \mid u_s)}}_{\textstyle i(z^\star_s;\,y_s \mid u_s)}\;} \tag{2}$$

Taking expectations over the generative process,

$$\mathbb{E}\big[\ln p_{\text{env}}\big] \;=\; \mathbb{E}\big[\ln \hat q_s\big] \;-\; I(z_s;\,y_s \mid u_s) \tag{3}$$

**Using the generator's own log-probability overestimates $\ln p_{\text{env}}$ by
exactly the mutual information between the generator's private prefix and its
public post, conditioned on what everyone can see.** The bias is non-negative
because mutual information is, and it is zero if and only if the post is
conditionally independent of the private prefix.

### 4.1 Three corollaries worth stating

**(a) The naive substitution is exact when there is no private prefix.** If the
generator opens `<post>` immediately, $z_s$ is empty, $I = 0$, and $\hat q_s$
*is* $p_{\text{env}}$. B6 measures the pre-post region at 0–2 tokens at early
checkpoints on every substrate
([`runs/_tier_b/b6_token_accounting/NOTES.md`](runs/_tier_b/b6_token_accounting/NOTES.md)),
so early in training the correction is not merely small — it is nil.

**(b) The naive substitution is never worse than dropping the term.** The dropped
term contributes $\mathbb{E}[-\ln p_{\text{env}}] = H(y_s \mid u_s)$ per span,
while the substitution's bias is $I(z_s;y_s\mid u_s) \le H(y_s\mid u_s)$, with
equality only if $y_s$ is a deterministic function of $z_s$. So substituting
strictly dominates dropping, and dominates it by a wide margin whenever the
private prefix is short.

**(c) The latent is exactly the region B6 measures.** $z_s$ is everything the
generator emitted that the observer does not receive — private prose, the
`<post>` tags themselves, and any trailing content. That is precisely the
non-post region. So `thinking_frac` is a **direct proxy for how much latent there
is**, and therefore for how large the correction can be. It rises from ~0 to
0.4–0.9 over training, which predicts a correction that starts at zero and grows.

## 5. What the uncorrected substitution does to the gradient

The observation term in the loss is $\sigma_s(E_s - M_s)$ with
$E_s = \ln p_{\text{env}}$ and $M_s = \ln p_{\theta_i}(y_s \mid c_i, x^i_{<s})$.
Substituting $\hat q_s$ for $E_s$ gives, by (2),

$$(\ln \hat q_s - M_s) \;=\; (E_s - M_s) \;+\; i(z^\star_s; y_s \mid u_s)$$

The extra term enters the REINFORCE return
$R_{t} = \sum_{s>t}\sigma_s(E_s - M_s)$ as a **future-dependent** additive
quantity, so unlike a history-determined baseline it is *not* annihilated by the
score-function identity. Minimising the loss therefore also minimises
$\sum_{s>t} i_s$, which is an incentive to take actions that **make other agents'
posts less dependent on their own deliberation**.

This is a different pathology from the one the dropped term creates, and a
stranger one. It is bounded by $I \le H(y\mid u)$ and is nil early, so it is
probably second-order — but it should be measured, not assumed away, and it is a
reason to prefer the marginalised estimator over the naive one where the private
region is large.

## 6. Tokenisation: work at the string level

The generator and the observer do not tokenise the same string identically. The
environment extracts the post text, strips and joins it, and the observer's
context re-encodes it inside different surrounding text; BPE then places
different boundaries. B5 measures this directly: the delivered string appears
verbatim in every observer's context (100%), while the generator's own token span
matches in only 42–88% of cases, degrading as post formatting degrades
([`runs/_tier_b/b5_env_likelihood/NOTES.md`](runs/_tier_b/b5_env_likelihood/NOTES.md)).

The resolution is not to align tokens. Both $\ln p_{\text{env}}$ and $M_s$ are
approximations, from opposite sides, to log-probabilities of **the same string**;
each summed under its own canonical tokenisation, each a lower bound on the
tokenisation-marginalised string probability. So:

> Compute both terms over the same **string span**, each under its own
> tokenisation, and treat $E_s - M_s$ as a per-string log-ratio. Never attempt
> per-token alignment, and never normalise the two sides by different token
> counts without saying which denominator is used.

This makes the per-token form of the objective a convenience rather than a
definition, which it always was.

## 6a. Private context is persistent, so the latent accumulates

Everything above treats $z_s$ as the private prefix of the turn that produced
$y_s$. That is too generous. **An agent's context persists across its turns
within an episode**, so the private text it generated at turn 1 is still
conditioning what it emits at turn $t$. The latent is not $z_t$ but the whole
private history

$$Z_t = (z_1, z_2, \ldots, z_t)$$

The identity (2) survives unchanged with $z^\star \to Z^\star_t$, and so does the
interpretation — the correction is still $I(Z_t; y_t \mid u_t)$ — but its *size*
now grows with $t$, and the estimator in §7 no longer works. Drawing a fresh
$z^{(k)}$ for the current turn only is a sample from the wrong distribution: the
prior that matters is $P(Z_t \mid u_t)$, and a valid draw requires resampling the
agent's entire private history conditioned on every post it has since made
public.

That is a sequential Monte Carlo problem — particles over private histories,
reweighted by each observed post — and over the ~16 turns of a forum episode the
weights degenerate. **So the correction is not merely larger when there is
private thinking; it stops being estimable at all.**

This is why the experiment in §9a removes the private channel rather than
modelling it. With no discarded tokens, $Z_t$ is empty at every turn, the
correction is identically zero, and $p_{\text{env}}$ is not estimated but
*computed*.

## 7. The estimator (only valid when the private prefix does not persist)

For each delivered post $y_s$ by generator $j$:

1. Reconstruct $u_s$ formatted as $j$ saw it, with $j$'s private prefix removed.
2. Draw $K$ fresh private prefixes $z^{(k)} \sim p_{\theta_j}(\cdot \mid u_s, c_j)$
   — generate under $j$'s adapter until it opens `<post>`, or to a cap.
3. Score $\ell_k = \ln p_{\theta_j}(y_s \mid u_s, c_j, z^{(k)})$.
4. $\ln \hat p^{(K)}_{\text{env}} = \operatorname{logsumexp}_k \ell_k - \ln K$.
5. Bias estimate $\hat\beta_s = \ln \hat q_s - \ln \hat p^{(K)}_{\text{env}}$,
   which estimates $i(z^\star_s;y_s\mid u_s)$ and whose mean estimates $I$.

$\hat p^{(K)}$ is unbiased for $p_{\text{env}}$; $\ln \hat p^{(K)}$ is downward
biased by $O(1/K)$ through Jensen, so $\hat\beta$ is an **upper** estimate of the
correction and small $K$ errs toward overstating it. Draw the $z^{(k)}$ from the
prior, not including $z^\star$, or the estimate is biased toward the realised
sample.

**This estimator is only correct for a single turn's latent.** By §6a the real
latent is the accumulated private history, so on a substrate with private
thinking the procedure above understates the correction by whatever the earlier
turns' private text still contributes. Treat it as a lower bound there, and as
exact only where the private channel is absent.

**Cost.** By §3 the estimate does not depend on the observer, so it is computed
once per post rather than once per (post, observer) pair. A 16-agent episode has
16 posts; at $K = 8$ that is 128 generate-and-score passes per episode, against
$16 \times 15 = 240$ pairs if it were observer-specific. All adapters share the
frozen base and are swapped in during rollout anyway.

## 8. What the corrected objective does to the dark room

With $E_s$ restored, the per-span term has expectation

$$\mathbb{E}_{y\sim p_{\text{env}}}\big[\ln p_{\text{env}}(y\mid u) - \ln p_{\theta_i}(y \mid c_i, u)\big] \;=\; \mathrm{KL}\big(p_{\text{env}} \,\Vert\, p_{\theta_i}\big) \;\ge\; 0$$

so the return becomes a **proper divergence**: non-negative, zero exactly when
the observer models the generator. Two consequences, one good and one that should
be stated plainly.

**The dark-room gradient is removed.** Under the surrogate, $-G_t = \sum_s M_s$
is maximised by making the future intrinsically low-entropy — the incentive the
rebuttal identified and B4 measures as the closing public channel. Under the
corrected return, driving the environment toward determinism drives
$p_{\text{env}} \to 1$ as well, and the term goes to zero from *both* sides. There
is no longer anything to gain by silencing other agents.

**But the corrected objective does not push *away* from the dark room either.** A
trivial world that is modelled perfectly and a rich world that is modelled
perfectly both give zero. The optimum is flat across worlds, so at convergence
advantages vanish and the action gradient self-extinguishes — whereas the
surrogate's does not, because $-G$ still ranks futures by their entropy. That may
be part of why the surrogate produces policy movement at all. Restoring the term
buys correctness and may cost the learning signal; that is an empirical question,
and it is the main risk of the corrected arm.

## 9. What to measure, and what would falsify this

Pre-registered in [`scripts/tier_b/c7_env_likelihood_bias.py`](scripts/tier_b/c7_env_likelihood_bias.py).

| prediction | what would falsify it |
|---|---|
| $\hat\beta \approx 0$ at early checkpoints | a large correction where the pre-post region is 0–2 tokens would mean the latent is not the non-post region and §4.1(c) is wrong |
| $\hat\beta$ grows with pre-post length, largest on `inner_loop` and `margin_notes` late (232 and 162 tokens), smaller on `ashbourne_gc` (35) | no relationship would mean the private prefix is not what the post depends on |
| $\hat\beta \le$ the per-span entropy $H(y\mid u)$ | a violation would indicate an estimator or alignment error, since (b) is a theorem |
| the correction is the same for every observer of one post | any observer dependence in the forum substrate contradicts §3 and would point at a hidden channel we have not accounted for |

**Stratify by whether the private region is degenerate.** B6 finds 8–11% of late
generations have token-loop private regions. A loop is highly predictive of more
loop, so those spans will show large $I$ for an uninteresting reason. They should
be reported separately or the aggregate will be misleading.

## 9a. The experiment: two runs with the private channel removed

The corrections above are all consequences of tokens the environment discards.
Remove the discarding and they all vanish at once. Configs in
[`configs/env_likelihood_bias/`](configs/env_likelihood_bias/).

**Substrate.** `study_group_ashbourne_gc`, 8 agents, Qwen2.5-7B-Instruct — the
campaign's reference setting, with the post tags blanked. Everything an agent
generates is its post; the prompt no longer offers a private space; the whole
384-token budget is postable. There is no `<python>` tool (the env refuses it in
this mode, since a code block would have nowhere to hide).

Then $Z_t$ is empty at every turn and, by §4.1(a),

$$p_{\text{env}}(y_t \mid u_t, j) = p_{\theta_j}(y_t \mid u_t, c_j) \quad\text{exactly.}$$

Better than exact — it is **already computed**. The generating agent's sampled
per-token log-probs are stored at rollout, and the environment splices its exact
token ids into every observer's context, so each observation token carries its
generator's log-prob with no re-encoding and no alignment step.

One subtlety about *which* distribution that is. The stored log-probs come from
the sampling engine — vLLM at fp8 — while the training forward runs bf16. For
$p_{\text{env}}$ the sampled value is the right one, not an approximation to be
tolerated: the environment is defined by what actually generated the token, and
what actually generated it was the fp8 engine. A re-score under the training
weights would return the log-prob of a distribution that emitted nothing. The
consequence is that $E_s$ and $M_s$ sit on slightly different numerical footings,
since $M_s$ is a bf16 forward. The gap between the engines at identical tokens
therefore sets the resolution at which any $E_s$ difference can be read, and it
is logged (`envlp/sampling_vs_training_lp_{mean,absmean,std}`) so the floor is
measured rather than assumed. It is *not* a staleness effect: staleness needs
`async_rollout`, which these configs pin off (§9a, pathway).

**Arm A — surrogate return, environment term measured.** Trains exactly as every
run in this project has, on $-G_t$. $E_s$ is logged throughout, so the corrected
return, the dropped term's share, and the disagreement between the two returns
are all recorded per iteration without affecting training.

**Arm B — corrected return.** Trains on $R_t = \sum \gamma^{s-t}(E_s - M_s)$,
with the value head fitted to that return rather than to $-G_t$.

Both arms carry a **probe head** fitted to the environment part of the return
alone. Its input is detached, exactly as the main value head's is, so it trains
only its own parameters. It supplies the history-conditioned mean needed to
centre $E_s$, without which the gradient-bias covariance of §5 cannot be formed.

**What is logged every iteration.** `envlp/return_surrogate_mean`,
`envlp/return_env_mean`, `envlp/return_corrected_mean`,
`envlp/env_share_of_return`, `envlp/return_rank_corr`, `envlp/return_pearson`,
`envlp/probe_loss`, `envlp/env_resid_mean`, `envlp/env_resid_std`,
`envlp/bias_corr_adv_resid`, `envlp/bias_rank_corr_adv_resid`,
`envlp/obs_tokens_measured_frac`, `envlp/misaligned_observations`,
`envlp/provenance_misses`.

The rank correlation is the decision-relevant one: **if the two returns rank
actions the same way, the bias does not matter for learning however large it
is.** The magnitude questions are answered by the share and the residual.

**Scale of the effect available to be seen.** Other agents' post content is
about **32%** of an observer's context in a healthy 8-agent run and **73%** at 16
agents; $E_s = 0$ on all remaining scaffolding. So an 8-agent run bounds the
correction to roughly a third of the observation stream before it starts. Read
as a bound on how large `env_share_of_return` can get.

**Pre-registered.** `envlp/obs_tokens_measured_frac` must be 1.0 and both
`misaligned_observations` and `provenance_misses` must be 0; anything else means
the provenance path is broken and the run's numbers are void. Arm B is expected
to drift toward the behaviour of perception-only training as its world model
improves and the corrected return goes to zero (§8) — if it ends at low KL from
base with entropy intact, that is the flat-optimum prediction confirmed, not the
objective working, and the two must not be conflated.

## 10. Summary

1. $p_{\text{env}}$ is computable in this substrate class. Everything except
   other agents' posts is deterministic; other agents' posts are generated by
   adapters we hold.
2. The observer's conditioning is **not** an obstacle. Under full observability
   $p_{\text{env}}$ is observer-independent, because an agent's own private state
   is conditionally independent of what others emit given the public thread.
3. The one adjustment needed is marginalising the **generator's** private prefix,
   and the exact correction is $I(z;y\mid u)$ — non-negative, zero when the
   generator emits no private tokens, and bounded above by the term currently
   being dropped.
4. Compare at the string level; per-token alignment is neither available nor
   needed.
5. Restoring the term removes the incentive to silence other agents but flattens
   the optimum, and that trade-off is the thing to test.
