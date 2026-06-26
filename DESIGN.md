# Phase 1: Offline IQL Critic and Test-Time Q-Guided Action Sampling

## Overview

This phase adds **offline value estimation and test-time action guidance** to a pretrained LingBot-VA world-action model.

The base model is first trained normally on the target task distribution using its original video and action flow-matching objectives. It is then frozen. A small double-Q critic and value head are trained on top of frozen action-branch representations using offline IQL and configurable rewards.

At inference, the critic does not update the base policy. Instead, its gradient with respect to the predicted clean action chunk is added to the action flow velocity during sampling. The goal is to improve generated action chunks while preserving the pretrained policy as a strong behavior prior.

This phase intentionally excludes video-side Q learning and video-flow guidance.

---

## Scope

### Included

- Task-SFT/pretraining of LingBot-VA using the existing training pipeline.
- Frozen LingBot-VA backbone during critic training.
- Offline chunk-level IQL critic training.
- Sparse terminal success reward and optional precomputed JEPA delta-distance reward.
- Test-time Q-guided action flow sampling.
- Action-side critic attached to frozen action-DiT hidden states.

### Excluded

- Video-fidelity reward or video-side Q critic.
- Q-guidance of video generation.
- Joint RL fine-tuning of the LingBot backbone.
- Online rollout collection or online policy updates.
- Learned dense progress reward.

These may be added only after Phase 1 is stable and evaluated.

---

## Base Model Assumptions

LingBot-VA generates autoregressive chunks:

$$
\text{history / language / observations}
\rightarrow
\text{video chunk}
\rightarrow
\text{action chunk}.
$$

The released model uses a unified transformer with modality-specific video/action input and output adapters. The action branch receives contextual information through the transformer, including history, language, and generated visual context.

For this phase, the action branch is treated as the control policy. The video branch remains unchanged.

---

## RL Formulation

One inference-sized latent video chunk and all of its aligned actions are treated as one RL action. Define:

$$
K = \texttt{infer\_latent\_chunk\_size},
\qquad
N = \texttt{action\_per\_frame}.
$$

The action tensor for transition $t$ is:

$$
A_t \in \mathbb{R}^{C_{\mathrm{action}} \times K \times N \times 1}.
$$

Thus one RL action contains $K$ latent frames and $K N$ aligned action tokens. An `action_config` entry is only a longer precomputed storage clip and may yield several such RL transitions.

The effective state includes all conditioning information available at chunk $t$, and the critic estimates $Q(s_t,A_t)$. The offline dataset is:

$$
\mathcal{D}
=
\{(s_t,A_t,r_t,s_{t+1},d_t)\}.
$$

Let $Z_t$ denote the current generated video chunk paired with action chunk $A_t$. The IQL state alignment is:

$$
s_t \leftarrow Z_{t-1},
\qquad
s_{t+1} \leftarrow Z_t.
$$

The action-branch hidden state for $Q(s_t,A_t)$ may use the current action/video chunk $(A_t,Z_t)$. The state-only value function must not use $Z_t$ for $V(s_t)$, because $Z_t$ already encodes information about the current action outcome.

For sparse-success training:

$$
r_t =
\begin{cases}
1, & \text{if this RL chunk reaches successful episode termination}, \\
0, & \text{otherwise}.
\end{cases}
$$

The discount is $\Gamma_t=\gamma^{H_t}$, where $H_t$ is the number of valid environment actions represented by the chunk. The training `infer_latent_chunk_size` must match inference `frame_chunk_size`.

For dense JEPA reward training, preprocessing computes dense patchwise cosine
distance from each state to a goal state:

$$
D_t
=
\mathrm{mean}_{c,h,w}
\left[
1-\cos
\left(
F_{t,c,h,w},
G_{c,h,w}
\right)
\right].
$$

Successful trajectories use their own final JEPA feature map as $G$. Failed
trajectories choose the closest successful final feature map as $G$. The
per-latent reward is progress toward the goal between adjacent latent states:

$$
r_j^{\mathrm{JEPA}}
=
D_j
-
D_{j+1}.
$$

The final latent reward is zero. For an inference-sized RL chunk, the transition
reward is the sum of the per-latent rewards inside that chunk.

Training may use either sparse success reward or JEPA delta reward. When sparse
success is included with JEPA reward:

$$
r_t
=
\alpha r_t^{\mathrm{JEPA}}
+
\beta r_t^{\mathrm{success}}.
$$

---

## Architecture

### Frozen LingBot Backbone

All pretrained LingBot parameters remain frozen during critic training:

- unified transformer;
- video and action adapters;
- video/action output heads;
- language and history conditioning modules.

The backbone provides contextualized action-token hidden states.

### Feature Layer Specification

Each critic checkpoint owns one versioned feature tap:

```text
feature_layers: [-1] or [block_index]
feature_aggregation: single
```

`feature_layers=[-1]` preserves the original behavior: action and video tokens
are taken after all DiT blocks and final adaptive `norm_out`, before the output
projections. A non-negative, zero-based block index selects the raw hidden
state immediately after that DiT block, before final adaptive normalization.
Q, online V, and target V must all use the same selected tap.

Phase 1 requires exactly one layer and `feature_aggregation=single`. The list
format reserves a stable checkpoint/config contract for later mixtures such as
mean, learned weighted sum, or concatenation. Such mixtures require explicit
aggregation and normalization versions; mixing final-normalized and raw block
features is invalid unless a common normalization is defined.

### Critic Input

Given a clean dataset action chunk $A_t$, run the frozen action branch at the clean endpoint using attention `chunk_size=K`, matching inference. Preserve:

$$
H_{\mathrm{act}}^\ell
\in
\mathbb{R}^{B \times K \times N \times d}.
$$

The critic requires one scalar per RL chunk. Masked-mean pool all valid action hidden states across both the latent-frame and action-per-frame axes:

$$
h_Q
=
\frac{\sum_{f=1}^{K}\sum_{i=1}^{N}m_{f,i}H_{\mathrm{act},f,i}^\ell}
{\sum_{f=1}^{K}\sum_{i=1}^{N}m_{f,i}}.
$$

This produces $h_Q \in \mathbb{R}^{B \times d}$, followed by one Q value per batch item. The same K, attention grouping, masks, and pooling must be used during training and guided inference.

### Double-Q Heads

Use two independent MLP heads:

$$
Q_{\psi_1}(s_t,A_t)
=
\mathrm{MLP}_{\psi_1}(\mathrm{LN}(h_Q)),
$$

$$
Q_{\psi_2}(s_t,A_t)
=
\mathrm{MLP}_{\psi_2}(\mathrm{LN}(h_Q)).
$$

Define:

$$
Q_{\min}(s_t,A_t)
=
\min(Q_{\psi_1}(s_t,A_t), Q_{\psi_2}(s_t,A_t)).
$$

The double critic reduces overestimation, which is particularly important because critic gradients will later steer generated actions.

### State-Value Head

IQL also requires a state-value function:

$$
V_\phi(s_t).
$$

For transition $t$, extract video tokens from the configured feature tap of the previous latent chunk $Z_{t-1}$:

$$
h_{V,t}
=
\mathrm{MaskedMeanPool}(H_{\mathrm{video}}^\ell(Z_{t-1})),
$$

then:

$$
V_\phi(s_t)
=
\mathrm{MLP}_{\phi}(\mathrm{LN}(h_{V,t})).
$$

The first episode chunk has no predecessor; exclude it from the expectile value loss while retaining its Q loss. For the Bellman target, apply the EMA target value head to current video tokens $Z_t$, which represent $s_{t+1}$.

---

## Offline IQL Training

### Value Loss

Fit the state-value function to an upper expectile of observed action values:

$$
L_V
=
\mathbb{E}_{(s,A)\sim\mathcal D}
\left[
\rho_\tau
\left(
Q_{\min}(s,A)-V_\phi(s)
\right)
\right],
$$

where:

$$
\rho_\tau(u)
=
|\tau-\mathbf{1}[u<0]|u^2.
$$

Initial setting:

$$
\tau = 0.7.
$$

This may be increased after observing the action-quality distribution in the dataset.

### Critic Target

Use an EMA target value network $V_{\bar\phi}$:

$$
y_t
=
r_t
+
(1-d_t)\Gamma_t V_{\bar\phi}(s_{t+1}),
$$

where $V_{\bar\phi}(s_{t+1})$ pools current video-chunk tokens $Z_t$. It does not load $Z_{t+1}$; doing so would shift the target one transition too far forward.

### Double-Q Loss

$$
L_Q
=
\mathbb{E}_{\mathcal D}
\left[
\sum_{i=1}^{2}
\left(
Q_{\psi_i}(s_t,A_t)-y_t
\right)^2
\right].
$$

### Total Loss

$$
L_{\mathrm{IQL}}
=
L_Q + \lambda_V L_V.
$$

Only critic and value-head parameters are updated:

$$
\nabla_{\theta_{\mathrm{LingBot}}}
L_{\mathrm{IQL}}
=
0.
$$

No actor/policy extraction loss is used. The pretrained LingBot action flow remains the reference policy.

---

## Test-Time Q-Guided Action Sampling

At action flow time $\tau$, LingBot samples:

$$
A_\tau \in \mathbb{R}^{B \times C_{\mathrm{action}} \times F \times N \times 1},
$$

where $F=\texttt{frame\_chunk\_size}=K$. For this repository's scheduler:

$$
\hat A_{\mathrm{clean}}
=
A_\tau-\sigma_\tau v_\theta(A_\tau,\tau).
$$

Feed the complete clean chunk through the critic. Pooling across its valid $F N$ hidden states produces one $Q_{\min}$ per batch item, so no per-latent Q reduction is required.

Compute:

$$
g_\tau
=
\nabla_{\hat A_{\mathrm{clean}}}
Q_{\min}(s_t,\hat A_{\mathrm{clean}}).
$$

Before using $g_\tau$, zero gradient entries for padded positions, invalid action channels, and server-clamped `action_cond` positions. These entries either do not represent an action or will be overwritten by the server.

Use denoising-step-aware scaling with time descending from $1$ to $0$:

$$
\rho(\tau)
=
\frac{\tau^2}{\tau^2 + (1-\tau)^2},
\qquad
s(\tau)
=
\min
\left(
\beta,
\frac{\tau}{(1-\tau)\rho(\tau)+\epsilon}
\right),
\qquad
\beta=2.
$$

Optional gradient normalization and clipping may be applied for stability. The guided clean-action estimate is:

$$
\hat A_{\mathrm{clean}}^{\mathrm{guided}}
=
\hat A_{\mathrm{clean}}
+
\lambda_{\mathrm{guide}} s(\tau) g_\tau.
$$

Since $\hat A_{\mathrm{clean}} = A_\tau-\sigma_\tau v_\theta$, the equivalent velocity correction used by the flow scheduler is:

$$
v_{\mathrm{guided}}
=
v_\theta
-
\frac{\lambda_{\mathrm{guide}}s(\tau)}{\max(\sigma_\tau,\epsilon)}
g_\tau.
$$

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

$$
\nabla_{\hat A_{\mathrm{clean}}}Q
$$

can be computed. The frozen action DiT forward pass must not be wrapped in `torch.no_grad()` during guided inference.

---

## Implementation Plan

### Stage 0 — Verify Model Interfaces

- Identify the action flow-time convention.
- Identify action-token positions in the unified transformer.
- Expose final normalized or one raw post-block action/video feature stream.
- Version the selected layer and normalization in critic checkpoints.
- Verify previous-video/current-action/current-video temporal alignment.
- Verify action chunk shape and chunk-to-environment transition alignment.

### Stage 1 — Dataset Construction

Build chunk-level offline transitions:

```text
previous_video_chunk -> V(s_t)
current_action_chunk -> Q(s_t, A_t)
current_video_chunk  -> target V(s_{t+1})
reward_t, done_t
```

Log task ID, instruction, episode ID, chunk index, predecessor validity, and success status for analysis.

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

$$
\text{task success rate}.
$$

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
infer_latent_chunk_size: 4  # must match inference frame_chunk_size
chunk_discount: gamma ** valid_environment_actions_in_chunk

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
    configured final-normalized or raw post-block action/video streams

Critic input:
    full LingBot context + candidate clean action chunk

Critic architecture:
    mean pool + LayerNorm + double MLP heads

Value state:
    previous video chunk for V(s_t)
    current video chunk for target V(s_{t+1})

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

## Phase 3: Predicted-Video Critic Training

`training_distribution` names the feature distribution used to train the
critic. Phase 1/2 use `clean_dataset`: video/action tensors are loaded directly
from the offline dataset and the critic sees clean dataset-conditioned hidden
states. Phase 3 uses `predicted_video_conditioned_action`: the current video
chunk is generated online by the frozen LingBot video flow, and Q is trained on
action hidden states from the action flow conditioned on that generated video.

The Phase 3 transition is:

```text
history real video/action chunks
        + current generated video chunk
        + current generated action chunk
        -> Q feature for current action

last real video chunk
        -> V(s_t)

current real video chunk
        -> target V(s_{t+1})
```

This means Phase 3 intentionally uses different feature sources for Q and V:

- Q is on the inference-like distribution: action tokens attend to real history
  plus the current generated video.
- V is a state baseline over real video states: current online V pools the last
  real chunk; the Bellman target pools the current real chunk.
- The first chunk has no previous real state, so its V loss is masked, while its
  Q loss is still trained.

The first Phase 3 implementation extracts Q and V from clean feature passes:

```text
phase3_q_feature_timestep = 0.0
phase3_v_feature_timestep = 0.0
```

These keys are explicit because later experiments may train critics from
partially denoised feature taps. The first experiment keeps them at zero.

The Phase 3 dense reward is online:

$$
r_t = -\alpha \cdot d_{JEPA}(\hat{x}_t, x_t)
$$

where $d_{JEPA}$ is the mean dense patchwise JEPA cosine distance between
decoded predicted frames and cached actual JEPA targets for the same latent
chunk. Optional sparse terminal success can be added separately, but the first
Phase 3 config uses only the negative JEPA distance.

For batched training, Phase 3 packs `previous real chunk + current chunk` along
the frame dimension and uses the existing training-style mask. During video
generation, the current video stream is noisy/generated and the previous video
stream is clean history. During action generation, the current action stream is
noisy/generated and the current video condition is the generated video, not the
ground-truth current video. The mask lets current action tokens attend to real
history and current generated-video condition tokens, while preventing access to
same-block current ground-truth video/action condition tokens. This avoids the
server's linear KV-cache loop and allows chunks to be parallelized in a batch.

Full video denoising is required for the first Phase 3 experiment:

```text
phase3_video_exec_step = -1
```

This disables LingBot's partial-video-denoise path for critic training.
