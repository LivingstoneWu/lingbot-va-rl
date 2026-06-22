# RL Critic Training Workflow

Phase 1 trains MC or IQL critics against a frozen LingBot-VA transformer. Use
`wan_va/rl/train_critic.py`; the regular `wan_va/train.py` remains the base
world-action model training pipeline.

## 1. Prepare the Dataset

Use an existing latent LeRobot dataset. No latent or Parquet conversion is
needed, but every episode in `meta/episodes.jsonl` must contain a boolean
`success` field. See `dataset_requirements.md` for the optional termination
fields and validation rules.

The base VA config selected below must point to this dataset and its matching
text embeddings, latent layout, camera keys, and action layout.

## 2. Select the Base VA Config

Set `base_config_name` to a key registered in `VA_CONFIGS` in
`wan_va/configs/__init__.py`. The selected base config supplies:

- `dataset_path`, camera keys, latent patch geometry, and action layout;
- `frame_chunk_size` and `action_per_frame`;
- the transformer parameter dtype and default pretrained transformer path.

Confirm that these values describe the dataset used to produce the stored
latents. To use a specific trained transformer checkpoint, set
`transformer_path` explicitly to its `transformer` directory. When it is null,
the trainer uses `<base_config.resume_from>/transformer` when available, then
falls back to `<wan22_pretrained_model_name_or_path>/transformer`.

## 3. Create the Critic Config

Copy `wan_va/configs/critic_phase1.example.json` to a run-specific JSON file.
The loader is strict: unknown keys cause an error.

Required run choices:

- `output_dir`: new run directory; artifacts are written under its `checkpoints/` subdirectory.
- `algorithm`: `"mc"` for twin-Q Monte Carlo return regression or `"iql"` for
  twin Q plus an expectile V head.
- `critic_type`: currently `"twin_mlp_v1"`.
- `infer_latent_chunk_size`: one RL action's number of latent frames. It must
  equal the base config's `frame_chunk_size`. The trainer reads
  `frame_chunk_size`, rejects a mismatch, and then uses
  `infer_latent_chunk_size` to partition the dataset and construct transformer
  inputs. This preserves the same action horizon used at inference.
- `feature_dim`: transformer hidden width returned by `return_features=True`;
  currently `3072` for the supplied model. Training fails early on mismatch.
- `feature_layers`: a one-element list selecting the shared Q/V feature tap.
  `[-1]` uses final adaptive-normalized action/video streams. `[k]` uses raw
  streams immediately after zero-based DiT block `k`. The trainer rejects
  negative values other than `-1`, out-of-range indices, and multiple layers.
- `feature_aggregation`: currently must be `"single"`. This list-plus-
  aggregation format reserves future multi-layer mixtures without changing the
  checkpoint/config shape.
- `window_size`: FlexAttention history limit in interleaved latent/action chunk
  positions. It is conceptually equivalent to inference `attn_window`, but the
  critic config does not inherit that base-config value. Each Phase 1 feature
  pass contains one RL chunk, so any value of at least `1` is effectively
  unrestricted; keep the default `64` unless multi-chunk inputs are introduced.

Model capacity:

- `hidden_dim` and `num_layers` configure the critic MLP heads.

RL objective:

- `gamma` discounts each low-level action; a chunk discount is
  `gamma ** (valid_frames * action_per_frame)`.
- `expectile`, `value_loss_weight`, and `target_ema_rate` affect IQL only.
- In IQL, Q uses current action tokens, online V uses the previous video chunk,
  and target V uses the current video chunk. The first chunk has no predecessor,
  so its V loss is masked while its Q loss remains active.

Optimization and runtime:

- `learning_rate` and `weight_decay` configure the critic optimizer; the frozen
  LingBot-VA transformer is not updated.
- `batch_size` is per rank. The global optimizer batch is `batch_size *
  world_size`; the critic trainer currently has no gradient accumulation.
- `num_workers` is per rank, so a run creates up to `num_workers * world_size`
  data-loader worker processes. Set it to `0` to load data in each rank's main
  process.
- `num_steps` is the final global optimizer-step target.
- `log_interval` and `save_interval` are measured in optimizer steps.
- `seed` initializes Python, NumPy, PyTorch, and distributed sampler ordering.
- `resume_from` is null for a new run or a critic checkpoint directory such as
  `.../checkpoints/checkpoint_00001000` when resuming.

The critic JSON overrides the selected base config's `batch_size`; it does not
use the base config's `gradient_accumulation_steps` or `load_worker` values.

Guidance parameters do not belong in this training config. They will be added
later to copied checkpoint configs by the evaluation pipeline.

## 4. Run Training

Run commands from the repository root. CUDA is required.

Single GPU:

```bash
PYTHONPATH=. python -m wan_va.rl.train_critic \
  --config wan_va/configs/critic_phase1.example.json
```

Multiple GPUs on one node:

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  -m wan_va.rl.train_critic \
  --config path/to/critic_run.json
```

The transformer is frozen. In distributed runs it is sharded with FSDP, the
dataset uses a distributed sampler, and critic gradients are averaged across
ranks.

## 5. Monitor and Resume

Rank 0 writes:

```text
<output_dir>/
  checkpoints/
    config.json
    loss.jsonl
    checkpoint_00001000/
      config.json
      manifest.json
      training_state.pt
```

`checkpoints/loss.jsonl` records step, time, total loss, Q loss, unweighted
value loss, gradient norm, and critic parameter norm at `log_interval`.
For MC, `value_loss` is zero. Checkpoints are written at `save_interval`
and once more at the final step.

Each checkpoint manifest records `feature_layers`, `feature_aggregation`, and
`feature_normalization`. Resume rejects a checkpoint whose feature tap differs
from the current training config.

To resume, create a new run config or edit the current one:

```json
{
  "output_dir": "./checkpoints/phase1_critic_iql_resume",
  "resume_from": "./checkpoints/phase1_critic_iql/checkpoints/checkpoint_00001000"
}
```

Keep the other fields from the complete config. Resume restores critic and
optimizer state and rejects incompatible checkpoint schema, algorithm,
`critic_type`, `feature_dim`, feature specification, or
`infer_latent_chunk_size` values. Set `num_steps` to the desired final global
step, not the number of additional steps.

## 6. Run Q-Guided Inference

Q-guided flow matching uses a separate server entry point:
`wan_va/wan_va_server_q_guiding.py`. The original `wan_va_server.py` remains
the unguided baseline. The guided server keeps the same reset, KV-cache, and
action response interface, and modifies only the action denoising loop.

Launch from the repository root:

```bash
PYTHONPATH=. python -m wan_va.wan_va_server_q_guiding \
  --config-name robotwin \
  --eval_model_path /path/to/lingbot-va/checkpoint \
  --q_checkpoint /path/to/critic/checkpoints/checkpoint_00010000 \
  --q_guidance_scale 0.1 \
  --q_guidance_beta 2.0 \
  --q_guidance_interval 5 \
  --q_objective min \
  --port 8000 \
  --robotwin
```

Use the same `--config-name`, `--eval_model_path`, `--port`, `--save_root`,
and `--robotwin` meanings as the unguided server. Q-specific arguments are:

- `--q_checkpoint`: critic checkpoint directory containing `config.json`,
  `manifest.json`, and `training_state.pt`.
- `--q_guidance_scale`: global strength of the Q-gradient correction. Use
  `0.0` to load the guided server but disable guidance.
- `--q_guidance_beta`: cap for the denoising-step-aware scale; default `2.0`.
- `--q_guidance_start_step` and `--q_guidance_end_step`: inclusive action
  denoising-step range where Q guidance is active. `-1` for the end step means
  no upper bound.
- `--q_guidance_interval`: apply Q guidance every N active action denoising
  steps. The default `1` guides every active step; larger values reduce the
  number of expensive Q feature-gradient passes.
- `--q_objective`: which twin-Q objective to maximize: `min`, `mean`, `q1`,
  or `q2`. Use `min` by default.
- `--q_grad_clip`: optional elementwise clip for the clean-action Q gradient;
  `0.0` disables clipping.
- `--q_grad_normalize`: optionally RMS-normalize the masked gradient before
  scaling.

The guided server validates that the critic checkpoint's
`infer_latent_chunk_size` equals the inference `frame_chunk_size`; when present,
the manifest `action_per_frame` must also match the base VA config. The critic's
`feature_layers`, `feature_aggregation`, and `feature_normalization` are read
from the checkpoint, so do not choose feature taps independently at inference.
Older schema-2 critic checkpoints trained before configurable feature layers
are accepted as legacy final-feature checkpoints and are treated as
`feature_layers=[-1]`, `feature_aggregation=single`, and
`feature_normalization=final_adaptive_norm_v1`.

The current QGF implementation evaluates Q on the clean action estimate:

```text
clean_action = noisy_action - sigma * predicted_velocity
```

It then applies a positive Q update in clean-action space. Because this
repository's flow velocity predicts `noise - clean`, the equivalent velocity
correction has a minus sign:

```text
guided_velocity = predicted_velocity - guidance_scale * scale * grad_Q / sigma
```

The denoising-step scale uses the configured scheduler sigma as the current
time value, descending from high noise to low noise:

```text
r_square = time**2 / (time**2 + (1.0 - time)**2)
scale = min(beta, time / ((1.0 - time) * r_square + 1e-8))
```

Guidance masks invalid action channels and first-frame `action_cond` positions
that the server clamps after every scheduler step.

## Preflight Checklist

- Every selected episode has `success` in `meta/episodes.jsonl`.
- The base VA config points to the intended latent dataset and embeddings.
- `transformer_path` identifies the intended frozen model checkpoint.
- `infer_latent_chunk_size` equals the base config's `frame_chunk_size`.
- `feature_dim` equals the transformer's returned hidden width.
- `feature_layers` contains one valid zero-based block index or `-1`.
- `feature_aggregation` is `"single"`.
- `output_dir` and `resume_from` identify the intended run.
- For Q-guided inference, `--q_checkpoint` matches the same action horizon and
  feature tap as the target inference server config.
- For non-final storage segments, latent frame count is divisible by the chunk
  size; the final segment covers `termination_frame` when provided.
