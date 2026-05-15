# A Guide to Training Agentic LLMs on Off-Policy Surprise Minimisation

## 0. Scope

This guide assumes familiarity with the Character-Conditioned Surprise Minimisation (CCSM) framework ([[Surprise Minimisation as a Unified Loss for Embodied LLM Agents]]) — the unified KL objective in which a single autoregressive model serves as world model, policy, and target, and behaviour is shaped by the framing prefix rather than an extrinsic reward. The on-policy algorithm (the §7.2 pseudocode of the derivation document) is the starting point. Here the question is narrower and practical: **how do we run this objective off-policy** — collecting rollouts under one set of parameters and updating under another — so that it trains efficiently on modern hardware, where keeping the rollout workers and the learner tightly synchronised is wasteful.

The headline result is encouraging. CCSM is _more_ replay-friendly than it first appears, because the expensive part of a training step (environment rollout) is cleanly separable from the cheap part (computing the return, which is a forward pass with no environment interaction). But there is one genuinely CCSM-specific complication that the standard off-policy toolkit (PPO, V-trace) does not address, and it is load-bearing: the perception pathway, which moves the self-referential target, has no importance-weighting story and is silently biased by stale trajectories. Most of this guide is about understanding why that matters and what to do about it.

---

## 1. Method overview

### 1.1 The on-policy objective in one paragraph

CCSM trains a single model $p_\theta$ by minimising the KL between the actual trajectory distribution (environment produces observation tokens, model produces action tokens) and a target distribution that is the model's own unconditional prediction of the whole trajectory. After collapsing the binary type mask $\sigma_t$ and dropping the intractable environment-entropy terms, the practical loss decomposes into two pathways sharing one forward pass. At observation tokens ($\sigma_t = 1$) the gradient is **next-token prediction** — the perception pathway, which trains the world model. At action tokens ($\sigma_t = 0$) the gradient is **REINFORCE** with intrinsic return $-G_{t'}$, where $G_{t'} = \sum_{s>t'}\sigma_s\big(-\ln p_\theta(x_s\mid x_{<s})\big)$ is the total surprise the model assigns to future observations. Actions that lead to predictable futures are reinforced; actions that lead to surprising futures are suppressed.

### 1.2 Why off-policy at all

On-policy training requires that every gradient step use rollouts collected under the current parameters. On modern hardware this couples the rollout workers and the learner: the learner idles while rollouts are generated, and the rollout workers idle while the learner steps. For LLM-scale agents, where a rollout involves many sequential environment turns and the policy is large, this coupling is the dominant inefficiency.

Off-policy training — or, more realistically, **off-policy with mild staleness** — decouples the two. Rollout workers run continuously and write trajectories into a buffer; the learner samples from the buffer. Trajectories may be a few iterations old by the time they are used. The cost is that the data no longer comes from the current policy, so the naive gradient is biased. The rest of this guide is about paying that cost correctly.

### 1.3 The two staleness problems

The single most important conceptual point in this guide: **CCSM has two distinct staleness problems, not one**, and they need different treatment.

**Problem 1 — trajectory distribution mismatch.** Rollouts were sampled under $\theta_{\text{old}}$ but the gradient is taken at $\theta$. This is the _ordinary_ off-policy problem, the one PPO and IMPALA were built to solve. Nothing about it is CCSM-specific.

**Problem 2 — the return is a function of $\theta$.** This one _is_ CCSM-specific and is easy to miss. In standard RL the reward $r_t$ is a fixed property of an environment transition: a stale trajectory still carries a perfectly valid reward. In CCSM the "reward" at observation token $s$ is $-\ln p_\theta(x_s\mid x_{<s})$ — the _current_ model's surprise. A trajectory collected at iteration $i$ has returns that are only correct if recomputed under the parameters you are actually updating. This is not a bug; it is the self-referential structure of the objective. But it dictates the replay design.

The good news about Problem 2 is that the fix is cheap and turns the self-referential structure from a liability into something the replay design naturally accommodates — see §2.1. The subtle news is that Problem 2 has a second-order consequence for the perception loss that §4 is entirely devoted to.

---

## 2. The core adjustments

### 2.1 Always recompute returns under current parameters

This is the first and most important change, and it is non-negotiable.

When a trajectory is pulled from the replay buffer, **do not** use any stored return. Recompute, under the current $\theta$ and current $V_\phi$:

- the per-token surprises $s_t = -\ln p_\theta(x_t \mid x_{<t})$,
- the returns $G_{t'} = \sum_{s>t'}\sigma_s,\gamma^{(s-t')} s_s$ (or a bootstrapped variant — see §3),
- the advantages $A_{t'} = -(G_{t'} - V_\phi(x_{\le t'}))$.

This is steps 2–4 of the on-policy algorithm, run again at replay time. It is a single forward pass of $p_\theta$ over the stored trajectory, with no environment interaction — cheap relative to the rollout that produced the trajectory.

Recomputing dissolves Problem 2 completely. The returns are always "on-policy-correct" with respect to the reward function, even when the trajectory itself is stale, because the reward function _is_ the current model and you have just evaluated it. What recomputation does **not** fix is Problem 1 — the observation tokens $x_s$ in the trajectory are still samples the current policy might not have induced. That is what §2.2 and §3 are for.

A useful way to hold this: recomputation makes the _reward function_ current; importance weighting makes the _trajectory distribution_ current. You need both, and they are separate operations.

### 2.2 Store behaviour log-probs and add PPO-style importance weighting

To correct Problem 1 on the action pathway, the learner needs to know the probability under which each action token was actually sampled. So the rollout worker must store, alongside each trajectory $(x_{1:T}, \sigma_{1:T})$, the behaviour-policy log-probs at action tokens:

$$\ln p_{\theta_{\text{old}}}(x_{t'} \mid x_{<t'}) \quad \text{for all } t' \text{ with } \sigma_{t'} = 0.$$

These are produced for free during rollout (the worker is sampling from $p_{\theta_{\text{old}}}$ anyway). At learner time, form the per-action-token importance ratio

$$\rho_{t'} = \frac{p_\theta(x_{t'} \mid x_{<t'})}{p_{\theta_{\text{old}}}(x_{t'} \mid x_{<t'})}$$

and replace the on-policy action loss with the PPO-clipped, importance-weighted form:

$$\mathcal{L}^{\text{act}} = -\sum_{t'} (1-\sigma_{t'}) \, \min\Big(\rho_{t'}\,\text{sg}(\hat A_{t'}),\; \text{clip}(\rho_{t'}, 1-\epsilon, 1+\epsilon)\,\text{sg}(\hat A_{t'})\Big).$$

This is exactly standard PPO. The only CCSM-specific content is what goes _inside_ $\hat A_{t'}$ — the recomputed, possibly bootstrapped advantage from §2.1 and §3 — not the clipping machinery itself.

### 2.3 The value head

The value head $V_\phi$ trains on a regression target $G_{t'}$ that, under off-policy training, is recomputed each replay. So $V_\phi$ is chasing a moving target for two compounding reasons: the self-referential shift of $\theta$ (already present on-policy) and now also the staleness of the trajectories it regresses on. Keep the stop-gradient on the regression target ($\text{sg}(G_t)$), as in the on-policy algorithm, and consider a slightly lower learning rate for $\phi$ than you would use on-policy, since its target is noisier. If $V_\phi$ shares the backbone with $p_\theta$, the existing stop-gradient-on-backbone advice for $L_{\text{val}}$ still applies and matters slightly more here.

---

## 3. Truncating the importance-weight horizon

### 3.1 Why the naive correction is a variance disaster

Section 2.1 noted that recomputing $G_{t'}$ fixes the reward function but not the trajectory distribution: the future observation tokens $x_s$ summed inside $G_{t'}$ were produced by the environment in response to _stale_ actions. A fully-correct off-policy Monte Carlo return would need a per-decision importance correction along the entire future — each future surprise term $s_s$ would be multiplied by the product of ratios for every action token between $t'$ and $s$:

$$\prod_{\substack{t' < u \le s \ \sigma_u = 0}} \rho_u.$$

This is the textbook per-decision importance sampling estimator, and it is the textbook variance disaster. The product of many ratios has variance that grows multiplicatively with horizon; for the long trajectories typical of agentic LLM rollouts it is unusable in practice. Do not implement the full product.

### 3.2 Option A — V-trace-style truncated weights

If you want to keep a multi-step Monte Carlo return, clip each ratio in the product at some $\bar\rho$ (typically $\bar\rho = 1$): replace $\rho_u$ with $\min(\bar\rho, \rho_u)$. This is the V-trace correction from IMPALA. It accepts a controlled bias in exchange for bounded variance, and it was designed for exactly the "actor–learner lag" regime — mild staleness — that is the target here. V-trace is the natural fit if you have an existing implementation; the only CCSM-specific adaptation is that the "rewards" being weighted are the recomputed surprises, not environment rewards.

### 3.3 Option B — bootstrap with the value head (recommended starting point)

The cheaper and, as a first move, preferable option: stop using the full Monte Carlo $G_{t'}$ and switch to a short bootstrapped return. An $n$-step return

$$G_{t'}^{(n)} = \sum_{t' < s \le t'+n} \sigma_s,\gamma^{(s-t')} s_s \;+\; \gamma^{n} V_\phi(x_{\le t'+n})$$

or a full GAE computation over the surprise stream truncates the importance-weight product to length $n$. Staleness then only has to be corrected over a short horizon, which keeps the variance of any residual weighting tractable. In the limit $n=1$ the product collapses to a single ratio and there is no product to worry about at all.

The derivation document already mentions GAE and a learned baseline as variance-reduction tools. The point to internalise is that **off-policy, bootstrapping changes status**: on-policy it is a nice-to-have for variance reduction; off-policy it is the mechanism that makes the importance correction tractable in the first place. Since $V_\phi$ is already in the algorithm, leaning harder on it is the single cheapest change that makes off-policy training viable.

The cost of bootstrapping is the bias that $V_\phi$ introduces — a worse value head means a more biased return. But that cost is already being paid on-policy; off-policy simply raises the stakes on $V_\phi$ quality.

### 3.4 Recommendation

Start with Option B at small $n$ (or $n=1$) and only reach for V-trace if you find the bootstrapped return is too biased — i.e. if $V_\phi$ is not good enough to carry the return and you need more real surprise signal in it. This keeps the first off-policy implementation as simple as possible.

---

## 4. The perception loss: the CCSM-specific risk

This section is the reason this guide exists. Everything in §2 and §3 is standard off-policy RL machinery wearing CCSM clothing. This is the part that is genuinely specific to the framework and is not covered by reaching for PPO.

### 4.1 The problem

The perception loss in the on-policy algorithm is plain next-token prediction on observation tokens:

$$\mathcal{L}^{\text{perc}}(\theta) = -\sum_{t} \sigma_t \ln p_\theta(x_t \mid x_{<t}).$$

It has **no importance-weighting term**. On-policy this is fine — the observation tokens are drawn from the current policy's state-visitation distribution, so NTP on them is unbiased for what we want. Off-policy it is _not_ fine: the observation tokens in a replayed trajectory were produced by the environment in response to the _old_ policy's actions. Training $\mathcal{L}^{\text{perc}}$ on them biases the world model toward the old policy's state-visitation distribution.

For ordinary LLM RL pipelines, people typically ignore exactly this kind of bias in the supervised-style component, and it is usually harmless. **For CCSM it is not safe to ignore by default**, because of what the perception pathway _does_.

### 4.2 Why it is load-bearing

Recall the stability argument for the self-referential KL (§3.3 of the derivation document). The objective has a moving target — both sides of the KL depend on $\theta$ — and the argument that this nonetheless converges rests on a **two-timescale separation**: the perception pathway (NTP, low-variance, supervised-like) moves the target _slowly and smoothly_, while the action pathway (REINFORCE, high-variance) moves the policy. The target moving smoothly relative to the noisy policy is what makes the fixed-target intuition approximately valid over short timescales and what makes stable convergence plausible.

The perception pathway is therefore not an incidental supervised loss bolted onto an RL objective. It _is the mechanism that moves the self-referential target_. If off-policy replay biases or destabilises that pathway, it is not degrading a side component — it is interfering with the thing the stability story depends on.

Heavy replay makes this worse in a specific way. If each trajectory is replayed many times, the perception loss is repeatedly trained on a stale, fixed trajectory distribution. The target stops tracking the _current_ policy's experience and starts tracking the _buffer's_ experience. The two-timescale separation does not break loudly; it erodes. The likely empirical signature is the perception loss curve and the self-consistency behaviour starting to diverge — the model getting good at predicting buffer-distribution observations while the policy has moved elsewhere.

### 4.3 Mitigations

In rough order of increasing effort:

**Keep replay shallow.** The simplest mitigation, and it aligns with the "off-policy with mild staleness" framing that motivated this whole exercise. Bound trajectory age (at most $N$ iterations old) and reuse (each trajectory used at most $M$ times), with small $N$ and $M$. If staleness is genuinely mild, the perception-loss bias is correspondingly mild and may need no further correction. This is the recommended default.

**Down-weight stale samples in $\mathcal{L}^{\text{perc}}$.** A scalar weight on the perception contribution of each trajectory that decays with the trajectory's age. Cheap, crude, and often enough.

**Importance-weight the observation tokens.** The principled fix: apply a per-token importance correction on observation tokens too. This is cheaper than it sounds — the forward pass needed to compute $p_\theta(x_t \mid x_{<t})$ at observation tokens is already being done for the perception loss, and the behaviour-policy probabilities at observation tokens are deterministic-environment-conditioned, so the ratio you actually need is over the _action_ tokens preceding the observation (the trajectory reached this observation because of stale actions). In practice this means weighting the perception loss at observation token $s$ by a (truncated) product of action ratios since the last few actions — the same truncation logic as §3. The cost is implementation complexity and the same variance considerations as the action pathway.

### 4.4 The ablation worth running

The severity of §4.1 is genuinely uncertain and depends on the relative timescales in a specific setup — how fast the self-referential target is moving, how stale the buffer is, how expressive the model is. It is not something to settle by argument. The clean experiment:

> Fix a staleness budget. Run training with and without observation-token importance weighting on $\mathcal{L}^{\text{perc}}$. Watch whether the perception loss curve and a self-consistency metric diverge.

If shallow replay alone keeps them together, the simple mitigation is sufficient and the principled fix is unnecessary complexity. If they diverge, the importance weighting earns its place. Either way the experiment tells you something the theory cannot.

---

## 5. The KL regulariser does some of the work for free

The CCSM objective already includes a KL penalty against the pretrained model $p_{\theta_0}$ (§6.1 of the derivation document), motivated there as preventing policy collapse and — more subtly — stabilising the self-referential dynamics by anchoring $\theta$ to a region where the target $\tau_\theta$ varies slowly.

For off-policy training there is a third role, partly free: anchoring $\theta$ near $p_{\theta_0}$ also bounds how far $\theta$ can drift from $\theta_{\text{old}}$ within a few iterations, since both are pulled toward the same anchor. That indirectly bounds the importance ratios $\rho$ — which is precisely the job a PPO trust region does.

The practical consequence: the off-policy machinery and the stability machinery are partly the _same_ machinery. You may find you can run with a looser PPO clip $\epsilon$ than usual, because the pretrained-anchor KL is already constraining drift. This is worth _testing_ rather than assuming — the degree of overlap depends on the KL coefficient — but it means the two mechanisms should be tuned together, not independently. A natural thing to fold into a coefficient sweep: vary the pretrained-KL coefficient and the PPO clip jointly and look at where ratios actually land.

---

## 6. Implementation checklist

In rough order of effort-to-payoff.

1. **Recompute returns and advantages under current $\theta$ on every replay.** Non-negotiable. One forward pass, no environment interaction. This is the change that makes CCSM's self-referential structure _compatible_ with replay rather than fighting it. (§2.1)
    
2. **Store behaviour log-probs at action tokens during rollout; add PPO-clipped importance weighting to $\mathcal{L}^{\text{act}}$.** Standard PPO; the only CCSM-specific content is what goes inside the advantage. (§2.2)
    
3. **Switch $G_t$ from full Monte Carlo to bootstrapped $n$-step or GAE returns.** You have $V_\phi$ already. Off-policy this goes from "nice variance reduction" to "the thing that truncates the importance-weight horizon and makes the correction tractable." Start at small $n$. (§3.3)
    
4. **Apply V-trace-style truncated weights** if you keep multi-step return structure and find the bootstrapped return too biased; skip if a short bootstrap is enough. (§3.2)
    
5. **Decide explicitly how to treat $\mathcal{L}^{\text{perc}}$ under replay.** Do not let this ride on the default. Options: keep replay shallow (recommended default), down-weight stale samples, or importance-weight observation tokens. This is the CCSM-specific risk and the thing the standard toolkit does not cover. (§4.3)
    
6. **Cap staleness explicitly** — trajectory age $\le N$, reuse $\le M$ — and treat $N, M$ as swept hyperparameters, not fixed constants. The right values depend on how fast the self-referential target is moving, which is empirical. (§4.3)
    
7. **Tune the PPO clip and the pretrained-KL coefficient together,** since they do overlapping work. Expect to be able to loosen the clip relative to a no-KL-anchor setup. (§5)
    
8. **Run the perception-loss ablation** (§4.4) once the pipeline is up: same staleness budget, with and without observation-token importance weighting, watching for divergence between the perception loss curve and a self-consistency metric.
    

---

## 7. Summary

CCSM is unusually replay-friendly because environment rollout (expensive) is cleanly separable from return computation (a cheap forward pass), and because recomputing returns under the current parameters is the correct move both for off-policy correctness _and_ for honouring the self-referential semantics of the objective. The standard off-policy toolkit — PPO clipping, importance weighting, V-trace, bootstrapped returns — transfers directly to the action pathway with essentially no conceptual modification.

The one part that needs genuine care, and is not covered by reaching for PPO, is the **perception loss**. It carries no importance-weighting story, off-policy replay silently biases it toward the buffer's state-visitation distribution, and — because the perception pathway is the mechanism that moves the self-referential target — that bias interferes with the two-timescale separation the stability argument depends on. Shallow replay may well be enough to keep this benign; whether it is, is an empirical question best settled by the §4.4 ablation rather than by argument.

The framing that motivated this exercise — off-policy _with mild staleness_ — is also the framing that makes the perception-loss risk most manageable. Mild staleness means mild perception-loss bias. The methods in this guide are about being able to push staleness as far as efficiency demands while knowing exactly which mechanism breaks first when you push too far.

---

## References

- Espeholt, L., Soyer, H., Munos, R., et al. (2018). IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. _arXiv:1802.01561_. — V-trace.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. _arXiv:1707.06347_.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. _arXiv:1506.02438_. — GAE.
- Hafner, D., Ortega, P. A., Ba, J., Parr, T., Friston, K., & Heess, N. (2022). Action and Perception as Divergence Minimization. _arXiv:2009.01791v3_.
- Borkar, V. S. (2008). _Stochastic Approximation: A Dynamical Systems Viewpoint_. Cambridge University Press. — two-timescale analysis.