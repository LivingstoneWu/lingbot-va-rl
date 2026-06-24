# RL Dataset Requirements

No latent or Parquet reconversion is required. Update each episode record in `meta/episodes.jsonl`.

## Required Field

- `success` (`bool`): `true` when the task completed successfully; otherwise `false`.

## Recommended Fields

- `termination_frame` (`int`): final valid frame of the trajectory.
- `termination_reason` (`str`): reason such as `task_success`, `task_failure`, or `timeout`.
- `truncated` (`bool`): `true` when collection ended without a natural task termination; otherwise `false`.

## Example

```json
{
  "episode_index": 42,
  "success": true,
  "termination_frame": 381,
  "termination_reason": "task_success",
  "truncated": false,
  "action_config": []
}
```

## Validation

- Every episode used for RL training must contain `success`.
- Existing episode fields, including `action_config`, must be preserved.
- The final available `action_config` segment must cover `termination_frame` when that field is present.
- Write valid JSON Lines: one complete JSON object per line.

## Optional JEPA Delta Rewards

For dense JEPA reward training, each `action_config` entry must include:

```json
{
  "reward_config": {
    "reward_source": "jepa_delta_distance",
    "distance_metric": "cosine",
    "goal_episode_index": 12,
    "goal_selection": "self_success",
    "latent_rewards": [0.1, -0.02, 0.0],
    "distance_to_goal": [1.2, 1.1, 1.12]
  }
}
```

`latent_rewards` must contain one scalar per latent frame in that action_config
entry. Critic training sums these rewards over each inference-sized RL chunk.
