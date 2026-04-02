# Multi-Agent LLM Training Framework Specification

> **Revision note:** This revision removes the Interaction Protocol abstraction (previously §2.4), the Observation and Information Architecture section (previously §4), and all observation filtering logic from the framework. The framework no longer concerns itself with what information flows between agents — that is entirely the environment's responsibility. If observation filtering, communication channels, structured debate protocols, or information asymmetry are needed, they are implemented as PettingZoo environment wrappers in a separate library. The framework sees only: each agent produces tokens (actions) and receives tokens (observations). The token type mask has exactly two values. This simplification also removes the context window construction strategies (full history, sliding window, summarised history) from the framework — these are either model-level concerns or environment wrapper concerns. The Agent Lifecycle section (previously §5.3) is removed as it describes environment-level or experiment-orchestration concerns rather than training framework concerns.

---

## 1. Purpose

A framework for training language models (and vision-language models) through multi-agent interaction in arbitrary environments. The framework's scope is the training loop: collecting trajectories from a PettingZoo environment, computing losses, and updating model parameters. Everything about the social structure of interaction — who sees what, turn order, communication protocols — is the environment's concern and is opaque to this framework.

The motivating training objective is Character-Conditioned Surprise Minimisation (CCSM), where agents learn both to perceive (predict observations) and to act (choose actions that lead to predictable futures) under a single loss derived from KL divergence minimisation. However, the framework is agnostic to the specific loss function. CCSM, RLHF, PPO with extrinsic reward, or any future objective should be expressible within the same interface.

---

## 2. Core Abstractions

The framework is organised around five core abstractions. Each is defined as an interface (abstract base class) with concrete implementations as subclasses. No abstraction makes assumptions about the internals of any other.

### 2.1 Agent

An agent wraps a model (or a view onto a shared model) and exposes the interface needed by the training loop: generating actions given observations, computing log-probabilities over trajectories, and updating parameters.

**Interface:**

- `act(observation) → action, metadata` — Given an observation (tokens, images, structured data), return an action and any metadata the loss function will need (e.g. log-probabilities, entropy).
- `evaluate(trajectory, mask) → log_probabilities` — Given a complete trajectory and a mask indicating which tokens belong to this agent, compute the log-probability the agent assigns to each token. Used for both the perception loss (evaluating observation tokens) and the action loss (evaluating action tokens under the current policy vs. the rollout policy).
- `get_parameters() → parameters` — Return the trainable parameters for this agent (may be a subset of a shared model's parameters).
- `character_prompt` — The natural language prompt defining this agent's identity, goals, and dispositions.

**Implementations (subclasses):**

- **IndependentAgent** — Wraps a standalone model. Full parameter independence from all other agents.
- **SharedBackboneAgent** — Multiple agents share a single base model. Each agent has its own adaptor weights (e.g. LoRA) and its own character prompt. The shared backbone's parameters may be frozen, jointly trained, or trained with gradient scaling.
- **SharedModelAgent** — Multiple agents are literally the same model, differentiated only by character prompt. No agent-specific parameters at all. Closest to the "society of thought" paradigm where a single model takes different perspectives.
- **ExternalAgent** — Wraps an API call or fixed policy (e.g. a frozen model, a human, a rule-based agent). Cannot be trained. Useful for evaluation, human-in-the-loop experiments, and asymmetric training setups where some agents are teachers.

### 2.2 Environment

The environment is a black box conforming to the PettingZoo AEC or Parallel API. The framework makes no assumptions about what happens inside it.

The framework's contract with the environment is minimal:

- The environment exposes agent identities and turn order (provided by PettingZoo).
- At each step, the environment provides an observation for the acting agent(s) and accepts an action.
- The environment signals episode termination.

Everything else — observation filtering, communication channels, partial observability, structured interaction protocols, information asymmetry — is implemented inside the environment or as PettingZoo wrappers around it. The framework does not know or care.

### 2.3 Loss Function

The loss function takes trajectory data and agent metadata and produces scalar losses and any auxiliary outputs (advantages, value estimates, intrinsic rewards) needed for gradient computation.

**Interface:**

- `compute_loss(trajectory, agent, mask, metadata) → loss, diagnostics` — Given a trajectory, the agent whose loss to compute, a mask partitioning the trajectory into this agent's actions vs. observations, and any metadata from rollout (stored log-probs, etc.), return the scalar loss and a diagnostics dict.
- `requires_value_function → bool` — Whether this loss needs a value baseline.
- `requires_old_log_probs → bool` — Whether this loss needs log-probabilities from the rollout policy (for importance sampling / PPO clipping).
- `trajectory_level → bool` — Whether the loss operates at the trajectory level (requiring full rollouts) or can be computed token-by-token (allowing streaming / online updates).

**Implementations:**

- **CCSMLoss** — The surprise minimisation objective. Perception loss on observation tokens (NTP cross-entropy), action loss on action tokens (REINFORCE or PPO with intrinsic return = negative future observation surprise). Requires a value function.
- **ExtrinsicRewardLoss** — Standard RL loss with reward provided by the environment. PPO or REINFORCE with environment-provided reward signal.
- **SFTLoss** — Supervised fine-tuning loss. Cross-entropy on all tokens or a specified subset. No RL component.
- **HybridLoss** — Weighted combination of any of the above. Supports mixing CCSM with extrinsic reward, or SFT with RL, with configurable coefficients that may change over training.
- **CustomLoss** — User-defined loss function conforming to the interface.

### 2.4 Tokeniser / Observation Encoder

Handles conversion between raw environment observations/actions and the token sequences that models consume and produce. This is the modality boundary.

**Interface:**

- `encode_observation(observation) → tokens` — Convert a raw observation (text, image, structured data) into a token sequence suitable for the model's input.
- `decode_action(tokens, action_space) → action` — Convert model-generated tokens into an action compatible with the environment's action space.
- `build_trajectory(episode_history) → token_sequence, type_mask, agent_mask` — Given the full history of an episode (all observations and actions with agent and turn metadata), produce the interleaved token sequence, a mask indicating token types (observation vs. action), and a mask indicating which agent produced each action token.

**Implementations:**

- **TextTokeniser** — For text-only environments. Wraps a standard tokeniser (the model's own). Actions and observations are already text.
- **VLMTokeniser** — For multimodal environments. Encodes images as vision tokens (using the VLM's vision encoder), interleaves them with text tokens, and produces a unified sequence.
- **StructuredTokeniser** — For environments with structured (non-text) observation and action spaces. Serialises structured data into a text representation (JSON, natural language description, or a learned encoding) before tokenisation.

### 2.5 Trajectory Store

Stores collected rollouts and serves them to the training loop. Handles the bookkeeping of which agent did what, what the stored log-probabilities were, what the computed advantages are, etc.

**Interface:**

- `store(trajectory, agent_metadata) → trajectory_id` — Store a completed trajectory with all associated metadata.
- `sample(batch_size, agent_id?) → batch` — Sample a batch of trajectories, optionally filtered by agent.
- `annotate(trajectory_id, key, value)` — Add computed quantities (advantages, returns, value estimates) to a stored trajectory.
- `clear()` — Flush the store (e.g. after a policy update in on-policy methods).

**Implementations:**

- **OnPolicyStore** — Simple buffer for on-policy methods (PPO, REINFORCE). Stores current rollouts, serves them once, then clears.
- **ReplayBuffer** — Off-policy buffer with configurable capacity and sampling strategy (uniform, prioritised). For methods that can reuse old trajectories.

---

## 3. Training Loop

The training loop orchestrates the interaction between the core abstractions. It is itself configurable but follows a standard structure.

### 3.1 Episode Collection

1. The environment resets. Each agent receives its initial observation.
2. Agents act in the order determined by the environment. At each step:
   - The acting agent receives its observation (as provided by the environment, unmodified) and produces an action plus metadata (log-probs, entropy).
   - The action is decoded and sent to the environment.
   - The environment produces the next observation(s).
   - All observations, actions, and metadata are recorded.
3. The episode terminates when the environment signals done or a maximum length is reached.
4. The tokeniser builds the full trajectory, type mask, and agent mask from the episode history.
5. The trajectory is stored in the trajectory store.

### 3.2 Loss Computation

1. For each agent, retrieve trajectories from the store.
2. Run a forward pass of the agent's model over each trajectory to compute current log-probabilities at all positions.
3. The loss function computes:
   - Per-token losses at relevant positions (determined by the mask and the loss function's requirements).
   - Any trajectory-level quantities: returns, advantages, value targets.
4. Losses are aggregated across trajectories in the batch.

### 3.3 Parameter Update

1. Gradients are computed from the aggregated loss.
2. For shared-backbone agents, gradients from all agents sharing the backbone are accumulated before the update step. The framework handles gradient routing: adaptor-specific gradients go to the relevant adaptor, backbone gradients are accumulated across agents (with optional per-agent scaling).
3. If a value function exists, its parameters are updated from the value loss.
4. Any learning rate schedules, gradient clipping, or other optimiser configuration is applied.

### 3.4 Iteration

The training loop repeats: collect episodes → compute losses → update parameters. The following should be configurable per training run:

- Number of episodes per collection phase.
- Number of optimisation steps per collection phase (for methods that do multiple passes over the same data, like PPO).
- Whether episode collection is synchronous (all episodes complete before any updates) or asynchronous.
- Whether agents are updated simultaneously or sequentially (relevant for non-stationarity management).
- Curriculum schedules: changes to the environment, the set of agents, or the loss function over training.

---

## 4. Agent Configuration

### 4.1 Weight Sharing Topologies

The framework supports arbitrary weight sharing between agents via the agent subclass hierarchy:

- **No sharing** — Each agent is a fully independent model. Maximum flexibility, maximum memory cost.
- **Full sharing, prompt-only differentiation** — All agents are the same model. The only per-agent parameter is the character prompt. Cheapest. Best for studying the effect of prompt/role variation in isolation.
- **Shared backbone with per-agent adaptors** — A single base model with lightweight per-agent adaptors. Balances specialisation with parameter efficiency. Adaptor architecture (LoRA, prefix tuning, adaptor layers, etc.) is a configuration choice.
- **Shared backbone with per-agent heads** — Shared feature extraction, but agent-specific output heads (e.g. separate value heads, separate policy heads for different action spaces).
- **Population-based** — A population of $M$ model variants, with $N$ agents sampled from the population for each episode. Supports evolutionary strategies over model parameters, prompts, or both.

### 4.2 Character Prompt Management

Character prompts are first-class objects in the framework, not just strings:

- **Static prompts** — Fixed for the duration of training.
- **Templated prompts** — Prompts with slots that are filled per-episode (e.g. "You are debating [TOPIC] and your position is [POSITION]").
- **Evolved prompts** — Prompts that are modified over training, either by a fixed schedule, by an outer optimisation loop, or by a prompt-evolution algorithm.
- **Hierarchical prompts** — A base prompt shared by all agents in a group, with per-agent extensions. Useful for defining team identity plus individual role.

---

## 5. Evaluation and Diagnostics

### 5.1 Per-Agent Metrics

For every agent at every evaluation point:

- **Observation surprise** — Mean negative log-probability on observation tokens. The core CCSM metric. Decomposable by source (environment tokens vs. other agents' tokens, if the agent mask makes this distinction available).
- **Action entropy** — Mean entropy of the policy at action positions. Measures exploration vs. exploitation.
- **Intrinsic return** — Mean discounted future observation surprise following the agent's actions. The CCSM reward signal.
- **Extrinsic return** — If the environment provides reward, the standard return.

### 5.2 Multi-Agent Metrics

- **Predictive asymmetry** — How well agent $i$ predicts agent $j$'s tokens vs. vice versa. Measures the epistemic power structure. Computed from the per-agent log-probabilities on tokens produced by specific other agents.
- **Behavioural diversity** — Distributional distance between agents' action distributions in matched contexts. Measures whether agents are maintaining distinct strategies or collapsing to conformity.
- **Coordination metrics** — Task-specific measures of whether agents are cooperating, competing, or ignoring each other. Defined per-environment.

### 5.3 Training Diagnostics

- **Non-stationarity tracking** — How rapidly each agent's observation distribution is changing due to other agents' policy updates. Measured as KL divergence between observation surprise distributions at successive evaluation points.
- **Gradient interference** — For shared-backbone agents: cosine similarity between gradients from different agents. Negative similarity indicates conflicting learning signals.
- **Loss decomposition** — For hybrid losses: contribution of each component (perception, action, value, entropy) to the total loss, tracked over training.
- **Collapse detection** — Automated monitoring for signs of mode collapse: action entropy falling below a threshold, observation surprise plateauing, behavioural diversity collapsing.

---

## 6. Curriculum and Scheduling

### 6.1 Environment Curriculum

The environment can change over training:

- **Fixed** — Same environment for all of training.
- **Staged** — A sequence of environments of increasing complexity.
- **Sampled** — Environments drawn from a distribution at each episode, with the distribution potentially shifting over training.
- **Adaptive** — Environment difficulty adjusts based on agent performance.

### 6.2 Social Curriculum

The social structure can change over training. Since social structure is determined by the environment (or its wrappers), social curriculum is implemented by swapping or reconfiguring environments:

- **Agent count scheduling** — Environments with increasing numbers of agents over training.
- **Protocol progression** — Swapping environment wrappers to increase interaction complexity over training (e.g. free-form → structured debate → adversarial debate).
- **Prompt evolution** — Character prompts change over training, either on a schedule or through optimisation.
- **Opponent sampling** — Which agents interact in each episode changes over training. Can be random, matched by skill, or structured to maximise learning signal.

### 6.3 Loss Scheduling

- **Loss annealing** — Coefficients on loss components change over training. E.g. start with high perception weight (learn the world model first), gradually increase the action weight (then learn to act).
- **Loss switching** — Change the loss function entirely at a training milestone. E.g. SFT warmup followed by CCSM.
- **Per-agent loss schedules** — Different agents may be on different loss schedules. E.g. a "teacher" agent on SFT while "student" agents train with CCSM against the teacher.

---

## 7. Extensibility Requirements

The framework must be straightforward to extend in the following ways without modifying core code:

- **New agent types** — By subclassing the Agent interface. Any model architecture, any parameter sharing topology.
- **New environments** — By implementing the PettingZoo AEC or Parallel API. No framework changes needed.
- **New loss functions** — By subclassing the Loss interface. Any combination of per-token and trajectory-level objectives.
- **New tokenisers** — By subclassing the Tokeniser interface. Any modality.
- **New evaluation metrics** — By registering metric functions that take trajectories and agent metadata and return scalar values.
- **New curriculum strategies** — By subclassing a Curriculum interface that controls how environment and loss evolve over training.

---

## 8. Configuration

A single training run is fully specified by a configuration object (or file) that defines:

- Which agents, with what weight sharing topology and character prompts.
- Which environment (a PettingZoo environment, possibly wrapped).
- Which loss function(s), with what coefficients and schedules.
- Which tokeniser / observation encoder.
- Training loop parameters: episodes per collection, optimisation steps per collection, batch size, max trajectory length.
- Evaluation schedule: how often to evaluate, which metrics to compute, which held-out environments or scenarios to evaluate on.
- Curriculum schedules for environment and loss.
- Logging and checkpointing configuration.

The framework should support experiment sweeps by parameterising any part of the configuration.

---

## 9. Non-Functional Requirements

- **Reproducibility** — Given the same configuration and random seed, a training run should produce identical results.
- **Experiment tracking** — All configurations, metrics, and checkpoints should be logged in a format compatible with standard experiment tracking.
- **Checkpoint and resume** — Training can be interrupted and resumed from any checkpoint, including the full state of all agents, the trajectory store, the curriculum position, and the random number generator state.
- **Scalability** — The framework should support single-GPU training for small experiments and scale to multi-GPU / multi-node for larger runs. The core abstractions should not assume any particular distributed training strategy.