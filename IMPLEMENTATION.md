# Phase 1 Implementation Plan: Offline Critics and Q-Guided Action Sampling

**Date:** 2026-06-19
**Status:** Planned
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
                +--> context-only features ----> V (IQL only)
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

### Chunk-Level Derived Fields

The RL dataset derives, rather than duplicates, sparse rewards:

```text
reward = 1 if success and this chunk reaches termination, otherwise 0
done = true if this chunk reaches termination, otherwise false
discount = 0 if done, otherwise gamma ** effective_chunk_duration
```

A failed terminal chunk has `reward=0` and `done=true`. It must never bootstrap into the first chunk of another episode. Timeouts remain distinguishable through `truncated`, even if the initial Phase 1 target treats them as terminal.

Each training sample should expose:

```text
episode_index
chunk_index
task/instruction
state inputs
action chunk and validity mask
reward
done
truncated
discount
next transition reference or next-state inputs
```

### Dataset Module Changes

Extend `wan_va/dataset/lerobot_latent_dataset.py` conservatively:

1. Preserve the existing supervised-training output by default.
2. Parse and retain episode outcome metadata in `new_metas`.
3. Add an RL transition wrapper or explicit RL mode instead of coupling Bellman-transition logic to normal flow training.
4. Build successor indices after sorting chunks within each episode.
5. Validate episode boundaries, chunk ordering, terminal alignment, and missing outcome labels.

Prefer a wrapper such as `ChunkTransitionDataset` under `wan_va/rl/transitions.py`. This keeps the existing dataset useful to both training pipelines and gives RL-specific behavior a narrow ownership boundary.

## Frozen Model Feature Interface

### Public Contract

Add `return_features: bool = False` to the relevant training/action forward paths in `wan_va/modules/model.py`.

Default behavior and return values must remain unchanged. When enabled, return predictions plus a structured dictionary:

```python
prediction, features = model(..., return_features=True)

features = {
    "action_tokens": action_tokens,
    "context_tokens": context_tokens,
    "action_mask": action_mask,
    "feature_spec": feature_spec,
}
```

Feature requirements:

- Capture final normalized transformer hidden states before `action_proj_out`.
- Preserve token masks for masked pooling.
- Clearly define which streams comprise `context_tokens`.
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

- Masked-mean pool final action tokens for `Q(s, A)`.
- Pool a verified context-only representation for `V(s)`.
- Apply layer normalization before each MLP head.
- Keep pooling and feature selection in versioned feature-extractor components.

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

IQL additionally trains `V(s)` from context-only features using expectile regression against a stopped-gradient `min(Q1, Q2)` target. Maintain an EMA target value head for Bellman targets.

If context-only feature extraction is not trustworthy in the first implementation session, complete the MC baseline before enabling IQL. Do not approximate `V(s)` using current-action tokens, because that changes the algorithm.

## Separate Critic Training Pipeline

Create a standalone entry point, proposed as `wan_va/rl/train_critic.py`. It may reuse checkpoint loading, distributed setup, logging, and dataset utilities, but it should not extend the flow-loss loop in `wan_va/train.py`.

Training responsibilities:

1. Load a pretrained LingBot-VA checkpoint.
2. Freeze all base-model parameters.
3. Build chunk transitions and balanced/suitable sampling over successful and failed trajectories.
4. Extract clean-action and context features.
5. Train only Q heads and, for IQL, V/target-V heads.
6. Log losses, calibration, rankings, success/failure separation, and return distributions.
7. Save critic state, optimizer state, resolved training config, and compatibility metadata.

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

The later inference implementation should use a guidance registry and validate critic/feature compatibility. Gradient guidance will enable autograd only for the candidate action path while keeping LingBot-VA parameters frozen.

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

- Add context-only value features and the V head.
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
- `return_features=False` preserves existing model behavior.
- Frozen backbone parameters receive no updates during critic training.
- Q values separate held-out successful and failed behavior better than trivial baselines.
- Twin critics do not exhibit uncontrolled disagreement or value scale growth.
- IQL context features exclude the candidate action.
- Checkpoint manifests reject incompatible base models or feature specifications.

Phase 1 is complete only after unguided and guided evaluation can be reproduced from a critic checkpoint plus an immutable experiment configuration and saved result directory.

