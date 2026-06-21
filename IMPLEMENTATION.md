# Phase 1 Implementation Plan: Offline Critics and Q-Guided Action Sampling

**Date:** 2026-06-19
**Status:** Critic training implemented; inference guidance planned
**Design reference:** `DESIGN.md`

## Concise Overview

Phase 1 adds offline action-value learning to a pretrained LingBot-VA model without updating the base model. The existing `wan_va/train.py` pipeline remains responsible for base world-action model training. A separate critic pipeline will freeze that checkpoint, extract action and context features through an optional `return_features=True` model interface, and train twin Q heads using either Monte Carlo (MC) return regression or Implicit Q-Learning (IQL).

Episode outcome metadata will be stored in the LeRobot `episodes.jsonl` records. The RL dataset layer will derive chunk-level transitions, sparse terminal rewards, termination flags, and successor references from those episode records.

Guidance is not part of critic training configuration. Later, the evaluation pipeline will copy the resolved training configuration into an experiment directory, append guidance and evaluation parameters, and save results there. This keeps checkpoints reproducible while allowing many guidance mechanisms and parameter sweeps to share one trained critic.

## Phase 1 Boundaries

### Included

- Episode success/failure metadata and chunk-level transition construction.
- Frozen LingBot-VA feature extraction.
- Twin-Q MC baseline.
- Twin-Q plus value-head IQL training.
- Critic checkpointing and compatibility metadata.
- Critic diagnostics and unit/integration tests.
- A later, separately scheduled action-guidance evaluation path.

### Excluded

- Updating LingBot-VA parameters with RL objectives.
- Online RL or rollout collection.
- Video-side critics or video-flow guidance.
- Dense learned rewards.
- Guidance configuration in the critic training config.
- Production deployment of guided inference.

## Overall Data Flow

```text
LeRobot episode data
  + episode success/termination metadata
                |
                v
chunk transition dataset
  (state, action, reward, done, next_state)
                |
                v
frozen LingBot-VA -- return_features=True
                |
                +--> action features ----------> Q1, Q2
                |
                +--> previous/current video features -> V/target-V (IQL only)
                                                   |
                                                   v
                                  versioned critic checkpoint

Later evaluation only:
critic checkpoint + copied training config + guidance config
                |
                v
guided inference experiment directory + evaluation results
```

## Dataset And Reward Schema

### Episode-Level Source Of Truth

Add outcome fields to each record in `meta/episodes.jsonl`:

```json
{
  "episode_index": 42,
  "tasks": ["place the cup"],
  "success": true,
  "termination_frame": 381,
  "termination_reason": "task_success",
  "truncated": false,
  "action_config": []
}
```

Required semantics:

- `success`: whether the task was completed.
- `termination_frame`: final valid frame in the trajectory.
- `termination_reason`: dataset-specific reason such as `task_success`, `task_failure`, or `timeout`.
- `truncated`: whether collection ended without a natural task termination.

The conversion/annotation tooling must validate that every RL episode has an outcome label. Missing labels should fail dataset validation rather than silently being treated as failures.

### Inference-Sized RL Chunks

One RL action is all actions aligned to one latent video chunk. Define:

```text
K = infer_latent_chunk_size
N = action_per_frame
A_t shape = [C_action, K, N, 1]
```

An `action_config` entry is only a stored/precomputed clip and may contain many latent frames. The RL transition view partitions its valid latent-frame dimension into consecutive K-frame chunks. Thus one storage clip with `F_total` latent frames can yield multiple RL transitions.

The training value of `infer_latent_chunk_size` must match inference `frame_chunk_size`. This avoids training Q on a different action horizon and attention pattern from the generated action chunk.

The RL dataset derives sparse rewards per K-frame chunk:

```text
reward = 1 if success and this RL chunk reaches episode termination, otherwise 0
done = true if this RL chunk reaches episode termination, otherwise false
discount = 0 if done, otherwise gamma ** valid_environment_actions_in_chunk
```

A failed terminal chunk has `reward=0` and `done=true`. A partial terminal chunk may be padded to K latent frames, but its latent and action masks must exclude padding from feature pooling and losses.

Each transition should expose:

```text
episode_index
storage_segment_idx
rl_chunk_idx
latent_start/end
latent/action tensors for K frames
latent and action masks
reward, done, truncated, discount
predecessor and successor transition references
state_valid (false only when no predecessor exists)
```

### RL Transition Wrapper

Keep `LatentLeRobotDataset` responsible for loading complete precomputed segments:

```text
latents:      [C_video, F_total, H, W]
actions:      [C_action, F_total, N, 1]
latents_mask: [F_total]
actions_mask: [C_action, F_total, N, 1]
```

Add `ChunkTransitionDataset` in `wan_va/rl/transitions.py` as a view over the base dataset. It will:

1. Parse episode outcomes retained in `new_metas`.
2. Order storage segments by episode and frame range.
3. Partition valid latent frames into K-frame RL chunks.
4. Assign stable transition indices and link each chunk to its previous and next chronological chunks in the same episode.
5. Derive reward, done, discount, predecessor validity, and masks.
6. Reject duplicate ranges, inconsistent outcomes, premature termination, and non-final segments that cannot be partitioned into full K-frame chunks.

The predecessor link supplies the video chunk used by online `V(s_t)`. The successor link remains useful for chronological validation and backward MC-return computation. For the first episode chunk, predecessor tensors are zero placeholders and `state_valid=false`, so only its value expectile loss is masked.

This wrapper leaves supervised flow training unchanged and keeps reward/Bellman semantics outside latent decoding.

## Frozen Model Feature Interface

### Public Contract

Add `return_features: bool = False` to the relevant training/action forward paths in `wan_va/modules/model.py`.

Default behavior and return values must remain unchanged. When enabled, return predictions plus a structured dictionary:

```python
prediction, features = model(..., return_features=True)

features = {
    "action_tokens": action_tokens,
    "video_tokens": video_tokens,
}
```

Feature requirements:

- Capture final normalized transformer hidden states before `action_proj_out`.
- Preserve the action layout as `[B, K, N, D]` for one RL chunk.
- Preserve latent/action masks for pooling across valid `K × N` tokens.
- Set the training attention `chunk_size` to `infer_latent_chunk_size`, matching inference `frame_chunk_size`.
- Return final normalized video tokens as `video_tokens`.
- Do not detach inside the model interface; the caller controls gradient behavior.
- Keep feature extraction numerically identical between critic training and guided inference.

### Clean-Action Feature Timestep

Critic training evaluates clean dataset actions at `action_feature_sigma=0.0`. The scheduler supports the clean endpoint, so epsilon is not required initially. If endpoint instability is demonstrated, a small configured sigma may be introduced and recorded in checkpoint metadata.

Use unambiguous names:

```text
action_feature_sigma: model feature-extraction noise level
iql_expectile: IQL expectile parameter, initially 0.7
```

For this scheduler, inference reconstructs the predicted clean action as:

```text
clean_action = noisy_action - sigma * predicted_velocity
```

## Critic Architecture

### Shared Feature Construction

- Reshape action hidden states to `[B, K, N, D]`.
- Masked-mean pool over both K and N, producing one state-action representation and one Q value per RL chunk.
- Pool previous-chunk `video_tokens` for online `V(s_t)`.
- Pool current-chunk `video_tokens` with target V for `V(s_{t+1})`.
- Apply layer normalization before each MLP head.
- Keep pooling, K, and feature selection in the versioned feature specification.

### Twin Q Heads

`Q1` and `Q2` are independently initialized networks estimating the same action value. They are not checkpoint or architecture versions.

```text
q1 = Q1(state_action_features)
q2 = Q2(state_action_features)
q_min = min(q1, q2)
```

Twin Q is used by both algorithms:

- MC baseline: both heads regress to the observed discounted episode return.
- IQL: both heads regress to `reward + discount * V_target(next_state)`.

The minimum reduces optimistic critic errors, especially when gradients later search for actions with high predicted value.

### IQL Value Head

IQL trains online `V(s_t)` from pooled video tokens of the previous latent chunk and applies the EMA target V to pooled video tokens of the current latent chunk for `V(s_{t+1})`. The current action hidden state remains the Q input.

The first episode chunk has no previous video chunk. Keep its Q loss, but mask its expectile V loss with `state_valid=false`. Never use current video tokens for online `V(s_t)`, because they already encode information about the current action chunk.

## Separate Critic Training Pipeline

The standalone entry point is `wan_va/rl/train_critic.py`. It reuses model loading and distributed FSDP support without extending the flow-loss loop in `wan_va/train.py`.

```bash
python -m wan_va.rl.train_critic \
  --config wan_va/configs/critic_phase1.example.json
```

Training responsibilities:

1. Load a pretrained LingBot-VA checkpoint.
2. Freeze all base-model parameters.
3. Build chunk transitions with explicit predecessor/successor links.
4. Extract current action/video features and previous video features.
5. Train Q from current action tokens, online V from previous video tokens, and target V from current video tokens.
6. Mask only first-chunk V losses while retaining their Q losses.
7. Log losses, calibration, rankings, success/failure separation, and return distributions.
8. Save critic state, optimizer state, resolved training config, and compatibility metadata.

The base model may require autograd during later guided inference with respect to candidate action inputs, but its parameters remain frozen. Critic training should avoid retaining base-model activation gradients when they are not needed.

## Training Configuration And Registry

The critic training config controls training only:

```yaml
training:
  algorithm: mc  # mc | iql
  critic_type: twin_mlp_v1
  feature_type: final_action_tokens_v1
  reward_type: sparse_terminal_v1
  action_feature_sigma: 0.0
  infer_latent_chunk_size: 4  # must match inference frame_chunk_size

dataset:
  path: ...
  gamma: ...

optimizer:
  learning_rate: ...
  weight_decay: ...
```

No guidance section belongs in this config.

Use registries for components with genuinely interchangeable behavior:

- critic architecture;
- feature pooling/extraction;
- reward derivation;
- target algorithm (`mc` or `iql`).

Each registered component owns and validates its configuration. Registries may accept new component-specific keys, but unknown component names and misspelled/unsupported keys must fail clearly. Avoid a single unvalidated dictionary that silently absorbs arbitrary settings.

## Checkpoints And Compatibility

Each critic checkpoint directory should contain:

```text
critic weights
optimizer/training state
resolved training config
training metrics
compatibility manifest
```

The manifest should record at least:

```text
checkpoint schema version
critic and feature component versions
algorithm (MC or IQL)
base-model checkpoint identity/hash
feature dimensions and token source
infer_latent_chunk_size and action_per_frame
chunk partitioning, padding, pooling, and predecessor/successor conventions
value-state alignment version
reward definition version
dataset fingerprint
action normalization statistics
action_feature_sigma
discount convention
IQL hyperparameters when applicable
```

Loading must validate compatibility before training resumes or evaluation begins.

## Later Guidance And Evaluation Pipeline

Guidance is an inference/evaluation concern and will be implemented after critic training is validated. It does not participate in critic optimization and is not stored in the original training config.

For each evaluation experiment:

1. Create a new experiment directory under or beside the critic checkpoint.
2. Copy the resolved training config from the checkpoint.
3. Append the selected guidance and evaluation parameters to the experiment copy.
4. Run evaluation without modifying the checkpoint's original config.
5. Save metrics, per-episode results, logs, and the fully resolved experiment config in that directory.

This supports multiple mechanisms and sweeps against one critic, for example:

```text
none
q_gradient_v1
candidate_rerank_v1
future advantage-based guidance
```

The later inference implementation should use a guidance registry and validate critic/feature compatibility. Inference samples one full action field with shape `[B,C,F,N,1]`, where `F=frame_chunk_size=infer_latent_chunk_size`. The critic pools the valid hidden states across `F × N` and returns one Q value per batch item, so no per-latent Q reduction is required.

Gradient guidance will enable autograd only for the candidate action path while keeping LingBot-VA parameters frozen. Before adding the gradient to the flow velocity, mask out:

- padded latent/action positions, which are not part of the RL action;
- invalid model action channels, which are placeholders for unavailable robot dimensions;
- `action_cond` positions clamped by the server, because they are overwritten after every scheduler step and are not controllable.

The fixed/clamped values may still be present as critic context; only their guidance update is zeroed.

## Proposed Module Layout

```text
wan_va/
  modules/model.py                 # optional return_features interface
  dataset/lerobot_latent_dataset.py
  rl/
    __init__.py
    transitions.py                # chunk transitions and reward derivation
    features.py                   # pooling and feature specifications
    critics.py                    # twin-Q and value heads
    algorithms.py                 # MC and IQL losses/targets
    registry.py                   # validated component construction
    checkpoint.py                 # manifests and compatibility checks
    train_critic.py               # standalone training entry point
    guidance.py                   # later evaluation phase
```

Configuration files should live with the existing configuration system unless repository conventions indicate a clearer dedicated `wan_va/configs/rl/` package during implementation.

## Implementation Sessions

### Session 1: Outcome Metadata And Transition Semantics

- Finalize the episode outcome schema.
- Add annotation/conversion validation.
- Implement and test chunk successor construction.
- Verify terminal reward and failure handling on representative episodes.

### Session 2: Feature Interface

- Add `return_features=True` without changing default outputs.
- Define action/context token boundaries and masks.
- Verify clean-action sigma behavior and tensor shapes.
- Test frozen-parameter and input-gradient behavior.

### Session 3: MC Twin-Q Baseline

- Implement feature pooling and twin MLP heads.
- Implement discounted MC targets.
- Add standalone training, checkpointing, and metrics.
- Confirm that values rank successful and failed trajectories meaningfully.

### Session 4: IQL

- Add previous/current video-state features and the V/target-V heads.
- Implement expectile value loss, EMA target V, and twin-Q Bellman targets.
- Compare MC and IQL calibration and ranking diagnostics.

### Session 5: Critic Hardening

- Add compatibility checks and resume tests.
- Test distributed/mixed-precision behavior.
- Audit success/failure balance and critic extrapolation.
- Freeze the first checkpoint schema.

### Session 6: Later Evaluation And Guidance

- Add experiment-directory creation and copied config augmentation.
- Implement guidance strategies independently of training.
- Add unguided controls, guidance sweeps, and per-episode result capture.
- Evaluate task success and action plausibility before considering later RL phases.

## Verification Gates

Do not proceed to guided inference until all of the following hold:

- Episode labels are complete and transition boundaries are correct.
- Terminal rewards and discounts have unit tests.
- `infer_latent_chunk_size` matches inference `frame_chunk_size`.
- Q pooling covers valid `K × N` tokens and returns one value per RL chunk.
- `return_features=False` preserves existing model behavior.
- Frozen backbone parameters receive no updates during critic training.
- Q values separate held-out successful and failed behavior better than trivial baselines.
- Twin critics do not exhibit uncontrolled disagreement or value scale growth.
- Online `V(s_t)` uses previous video tokens; target `V(s_{t+1})` uses current video tokens.
- First-chunk value losses are masked without removing their Q losses.
- Checkpoint manifests reject incompatible base models or feature specifications.

Phase 1 is complete only after unguided and guided evaluation can be reproduced from a critic checkpoint plus an immutable experiment configuration and saved result directory.

