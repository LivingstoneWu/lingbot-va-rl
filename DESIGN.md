# Phase 1: Offline IQL Critic and Test-Time Q-Guided Action Sampling

## Overview

This phase adds **offline value estimation and test-time action guidance** to a pretrained LingBot-VA world-action model.

The base model is first trained normally on the target task distribution using its original video and action flow-matching objectives. It is then frozen. A small double-Q critic and value head are trained on top of frozen action-branch representations using offline IQL and sparse terminal success reward.

At inference, the critic does not update the base policy. Instead, its gradient with respect to the predicted clean action chunk is added to the action flow velocity during sampling. The goal is to improve generated action chunks while preserving the pretrained policy as a strong behavior prior.

This phase intentionally excludes video-side Q learning and video-flow guidance.

---

## Scope

### Included

- Task-SFT/pretraining of LingBot-VA using the existing training pipeline.
- Frozen LingBot-VA backbone during critic training.
- Offline chunk-level IQL critic training.
- Sparse terminal success reward.
- Test-time Q-guided action flow sampling.
- Action-side critic attached to frozen action-DiT hidden states.

### Excluded

- Video-fidelity reward or video-side Q critic.
- Q-guidance of video generation.
- Joint RL fine-tuning of the LingBot backbone.
- Online rollout collection or online policy updates.
- Learned dense progress reward.
- Goal-image embedding / latent-distance reward shaping.

These may be added only after Phase 1 is stable and evaluated.

---

## Base Model Assumptions

LingBot-VA generates autoregressive chunks:

\[
\text{history / language / observations}
\rightarrow
\text{video chunk}
\rightarrow
\text{action chunk}.
\]

The released model uses a unified transformer with modality-specific video/action input and output adapters. The action branch receives contextual information through the transformer, including history, language, and generated visual context.

For this phase, the action branch is treated as the control policy. The video branch remains unchanged.

---

## RL Formulation

One action chunk is treated as one RL action:

\[
A_t = [a_t, a_{t+1}, \ldots, a_{t+K-1}].
\]

The effective state includes all conditioning information available at chunk \(t\):

\[
s_t =
(o_{\leq t}, a_{<t}, l, \text{video context/history}).
\]

The critic estimates:

\[
Q(s_t, A_t).
\]

The language instruction is part of the state representation. Therefore, the initial implementation does not require an explicitly separate goal input:

\[
Q(s_t, A_t)
\equiv
Q(o_{\leq t}, a_{<t}, l, \text{video context}, A_t).
\]

The offline transition dataset is:

\[
\mathcal{D}
=
\{(s_t, A_t, r_t^{(K)}, s_{t+K}, d_t)\},
\]

where \(d_t\) indicates terminal transition.

For Phase 1, use sparse terminal success reward:

\[
r_t^{(K)} =
\begin{cases}
1, & \text{if the chunk reaches successful termination}, \\
0, & \text{otherwise}.
\end{cases}
\]

The chunk-level discount is:

\[
\Gamma = \gamma^K.
\]

---

## Architecture

### Frozen LingBot Backbone

All pretrained LingBot parameters remain frozen during critic training:

- unified transformer;
- video and action adapters;
- video/action output heads;
- language and history conditioning modules.

The backbone provides contextualized action-token hidden states.

### Critic Input

Given a clean dataset action chunk \(A_t\), run the frozen action branch at the clean/data endpoint of the action flow schedule:

\[
H_{\mathrm{act}}^L
=
\mathrm{ActionDiT}_{\mathrm{frozen}}
(s_t, A_t, \tau_{\mathrm{clean}}).
\]

Here:

\[
H_{\mathrm{act}}^L \in \mathbb{R}^{K \times d}
\]

contains the final-layer hidden states for the \(K\) action tokens.

The exact numerical value of \(\tau_{\mathrm{clean}}\) must follow the repository’s flow-time convention. It should correspond to the data/clean-action endpoint, or the nearest timestep used during normal training.

### Pooling

The critic requires one scalar value per action chunk. Mean-pool the action-token hidden states:

\[
h_Q
=
\frac{1}{K}
\sum_{i=1}^{K} H_{\mathrm{act},i}^L.
\]

The transformer has already incorporated history, language, visual context, and the candidate action chunk before pooling. Mean pooling only reduces the action-token sequence into a single state-action representation.

### Double-Q Heads

Use two independent MLP heads:

\[
Q_{\psi_1}(s_t,A_t)
=
\mathrm{MLP}_{\psi_1}(\mathrm{LN}(h_Q)),
\]

\[
Q_{\psi_2}(s_t,A_t)
=
\mathrm{MLP}_{\psi_2}(\mathrm{LN}(h_Q)).
\]

Define:

\[
Q_{\min}(s_t,A_t)
=
\min(Q_{\psi_1}(s_t,A_t), Q_{\psi_2}(s_t,A_t)).
\]

The double critic reduces overestimation, which is particularly important because critic gradients will later steer generated actions.

### State-Value Head

IQL also requires a state-value function:

\[
V_\phi(s_t).
\]

Use a state-only representation extracted from the frozen transformer context/history tokens, excluding the current candidate action chunk. Let:

\[
h_V
=
\mathrm{MeanPool}(H_{\mathrm{context}}^L),
\]

then:

\[
V_\phi(s_t)
=
\mathrm{MLP}_{\phi}(\mathrm{LN}(h_V)).
\]

If clean extraction of context-only hidden states is difficult in the initial codebase, implement a Monte-Carlo-return critic baseline first, then add the IQL value head once the representation path is verified.

---

## Offline IQL Training

### Value Loss

Fit the state-value function to an upper expectile of observed action values:

\[
L_V
=
\mathbb{E}_{(s,A)\sim\mathcal D}
\left[
\rho_\tau
\left(
Q_{\min}(s,A)-V_\phi(s)
\right)
\right],
\]

where:

\[
\rho_\tau(u)
=
|\tau-\mathbf{1}[u<0]|u^2.
\]

Initial setting:

\[
\tau = 0.7.
\]

This may be increased after observing the action-quality distribution in the dataset.

### Critic Target

Use an EMA target value network \(V_{\bar\phi}\):

\[
y_t
=
r_t^{(K)}
+
(1-d_t)\Gamma V_{\bar\phi}(s_{t+K}).
\]

### Double-Q Loss

\[
L_Q
=
\mathbb{E}_{\mathcal D}
\left[
\sum_{i=1}^{2}
\left(
Q_{\psi_i}(s_t,A_t)-y_t
\right)^2
\right].
\]

### Total Loss

\[
L_{\mathrm{IQL}}
=
L_Q + \lambda_V L_V.
\]

Only critic and value-head parameters are updated:

\[
\nabla_{\theta_{\mathrm{LingBot}}}
L_{\mathrm{IQL}}
=
0.
\]

No actor/policy extraction loss is used. The pretrained LingBot action flow remains the reference policy.

---

## Test-Time Q-Guided Action Sampling

At action flow time \(\tau\), LingBot produces a reference velocity:

\[
v_\theta(s_t,A_\tau,\tau).
\]

Construct a local estimate of the clean action chunk:

\[
\hat A_{\mathrm{clean}}
=
A_\tau
+
\alpha(\tau)
v_\theta(s_t,A_\tau,\tau),
\]

where \(\alpha(\tau)\) follows the repository’s flow parameterization. For a linear interpolation convention with clean data at \(t=1\):

\[
\alpha(\tau)=1-\tau.
\]

Feed \(\hat A_{\mathrm{clean}}\) through the frozen action branch at the clean endpoint and evaluate:

\[
Q_{\min}(s_t,\hat A_{\mathrm{clean}}).
\]

Compute the action gradient:

\[
g_\tau
=
\nabla_{\hat A_{\mathrm{clean}}}
Q_{\min}(s_t,\hat A_{\mathrm{clean}}).
\]

Normalize or clip the gradient:

\[
\tilde g_\tau
=
\frac{g_\tau}
{\max(\|g_\tau\|_2,\epsilon)}.
\]

Then guide the action flow:

\[
A_{\tau+\Delta \tau}
=
A_\tau
+
\Delta\tau
\left[
v_\theta(s_t,A_\tau,\tau)
+
\lambda_{\mathrm{guide}}\tilde g_\tau
\right].
\]

The base flow preserves behavior-policy plausibility. The Q-gradient provides a conservative local preference toward higher predicted return.

### Gradient Rules

During critic training:

```text
Frozen LingBot parameters: yes
Gradient into LingBot: no
Trainable parameters: Q1, Q2, V heads only
```

During guided sampling:

```text
Frozen LingBot parameters: yes
Gradient through LingBot with respect to candidate action input: yes
Parameter update to LingBot: no
```

The action input must retain autograd so that:

\[
\nabla_{\hat A_{\mathrm{clean}}}Q
\]

can be computed. The frozen action DiT forward pass must not be wrapped in `torch.no_grad()` during guided inference.

---

## Implementation Plan

### Stage 0 — Verify Model Interfaces

- Identify the action flow-time convention.
- Identify action-token positions in the unified transformer.
- Expose final action-token hidden states.
- Expose context/history token hidden states for \(V(s)\).
- Verify action chunk shape and chunk-to-environment transition alignment.

### Stage 1 — Dataset Construction

Build chunk-level offline transitions:

```text
state_t
action_chunk_t
terminal_success_reward_t
state_t_plus_chunk
done_t
```

Log task ID, instruction, episode ID, chunk index, and success status for analysis.

### Stage 2 — Critic Baseline

Implement:

- action-token mean pooling;
- double-Q MLP heads;
- optional Monte-Carlo return regression baseline;
- critic ranking diagnostics.

Before Q-guided sampling, verify that:

- successful terminal chunks receive high values;
- failed chunks receive lower values, where failures are available;
- value estimates are bounded and stable;
- gradients with respect to actions are finite.

### Stage 3 — IQL Critic

Implement:

- value head;
- expectile value loss;
- twin Q TD loss;
- EMA target value network;
- checkpointing and critic diagnostics.

### Stage 4 — Guided Sampling

Implement:

- local clean-action estimate;
- critic evaluation at clean action endpoint;
- gradient extraction with respect to the action chunk;
- gradient normalization/clipping;
- configurable guidance scale.

### Stage 5 — Evaluation

Compare:

1. Base LingBot-VA.
2. Base LingBot-VA with zero guidance.
3. Q-guided LingBot-VA across multiple guidance scales.
4. Optional Monte-Carlo critic guidance versus IQL critic guidance.

Primary metric:

\[
\text{task success rate}.
\]

Secondary diagnostics:

- critic value distributions;
- action-gradient norms;
- action deviation from unguided policy;
- failure mode categories;
- success as a function of guidance scale.

---

## Initial Hyperparameters

These are initial defaults and must be validated experimentally.

```yaml
discount_gamma: 0.99
chunk_discount: gamma ** action_chunk_length

expectile_tau: 0.7
critic_lr: 3e-4
value_lr: 3e-4
target_ema: 0.005

q_hidden_dim: 512
q_num_layers: 2
use_layer_norm: true
double_q: true

guide_scale: [0.0, 0.01, 0.03, 0.1]
gradient_normalization: true
gradient_clip_norm: 1.0
```

The guidance scale is architecture- and action-normalization-dependent. It must be swept rather than assumed.

---

## Known Limitations

Sparse terminal success reward may be insufficient when:

- the offline data contains almost only successful demonstrations;
- there are few failed or suboptimal trajectories;
- action chunks have little variation at the same state;
- terminal success is far from early action chunks.

In this setting, the critic may behave more like a trajectory-progress estimator than a robust local action-value model. This is acceptable for the first experiment, but should be explicitly monitored.

Potential later extensions:

- latent-space progress shaping;
- learned task-conditioned progress model;
- goal relabeling;
- conservative critic regularization;
- action-branch LoRA fine-tuning;
- task-value gradients propagated into video-plan generation.

---

## Core Design Decisions

```text
Policy:
    frozen LingBot action flow

Critic location:
    final action-token hidden states

Critic input:
    full LingBot context + candidate clean action chunk

Critic architecture:
    mean pool + LayerNorm + double MLP heads

RL algorithm:
    offline IQL critic learning

Reward:
    sparse terminal success only

Policy improvement:
    test-time Q-guided flow sampling

Backbone update:
    disabled in Phase 1

Video-side RL:
    excluded in Phase 1
```