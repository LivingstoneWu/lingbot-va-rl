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

- `output_dir`: new directory for this run's config, logs, and checkpoints.
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
  `.../checkpoint_00001000` when resuming.

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
  config.json
  train.jsonl
  checkpoint_00001000/
    config.json
    manifest.json
    training_state.pt
```

`train.jsonl` records loss, Q means and disagreement, target mean, value loss,
and gradient norm at `log_interval`. Checkpoints are written at
`save_interval` and once more at the final step.

To resume, create a new run config or edit the current one:

```json
{
  "output_dir": "./checkpoints/phase1_critic_iql_resume",
  "resume_from": "./checkpoints/phase1_critic_iql/checkpoint_00001000"
}
```

Keep the other fields from the complete config. Resume restores critic and
optimizer state and rejects incompatible checkpoint schema, algorithm,
`critic_type`, `feature_dim`, or `infer_latent_chunk_size` values. Set
`num_steps` to the desired final global step, not the number of additional
steps.

## Preflight Checklist

- Every selected episode has `success` in `meta/episodes.jsonl`.
- The base VA config points to the intended latent dataset and embeddings.
- `transformer_path` identifies the intended frozen model checkpoint.
- `infer_latent_chunk_size` equals the base config's `frame_chunk_size`.
- `feature_dim` equals the transformer's returned hidden width.
- `output_dir` and `resume_from` identify the intended run.
- For non-final storage segments, latent frame count is divisible by the chunk
  size; the final segment covers `termination_frame` when provided.
