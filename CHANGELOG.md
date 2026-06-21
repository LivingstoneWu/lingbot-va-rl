# Changelog

## 2026-06-21 - Critic Training Workflow

### Added

- `workflow.md`: documents dataset preflight, base/critic config ownership, per-rank loader settings, attention-window semantics, launch commands, monitoring, and checkpoint resume behavior.

### Changed

- `DESIGN.md`: uses `$...$` for inline equations and `$$...$$` for display equations.

## 2026-06-20 - IQL Value-State Alignment

### Changed

- `ChunkRecord`: adds `previous_record_idx` while retaining successor metadata.
- `ChunkTransitionDataset.__getitem__`: returns optional `previous_*` tensors and `state_valid` for predecessor-aware IQL.
- `WanTransformer3DModel.forward_train`: renames `context_tokens` to `video_tokens`.
- `expectile_loss` / `iql_losses`: support masking unavailable first-chunk value states.
- `CriticTrainer._train_batch`: uses previous video tokens for online `V(s_t)`, current action tokens for Q, and current video tokens for target `V(s_{t+1})`.
- `CHECKPOINT_SCHEMA_VERSION`: advances to 2 for the incompatible value-state alignment.
- `DESIGN.md` / `IMPLEMENTATION.md`: explicitly define predecessor/current video semantics for IQL.
- Inference guidance remains unchanged.

### State Alignment

```text
previous video chunk -> online V(s_t)
current action tokens -> Q(s_t, A_t)
current video chunk  -> target V(s_{t+1})
no predecessor       -> mask V loss, retain Q loss
```

### Verification

- `PYTHONPATH=. pytest -q tests/test_rl_transitions.py tests/test_rl_critics.py tests/test_rl_config.py tests/test_rl_checkpoint.py`: 10 passed.

## 2026-06-19 - Phase 1 Offline Critic Training

### Added

- `ChunkTransitionDataset`: partitions latent storage clips into inference-sized RL chunks, links predecessors/successors, derives returns, and rejects non-final partial chunks.
- `ChunkRecord`: stores immutable chunk transition metadata.
- `MaskedTokenPool.forward`: pools valid action or video tokens across an entire latent chunk.
- `TwinQCritic`: evaluates independent Q1/Q2 heads from pooled action features.
- `ValueCritic`: evaluates chunk-level state values from pooled video features.
- `CriticBundle`: owns Q, V, target-V, and EMA target updates.
- `build_critic_bundle`: constructs registered critic versions and rejects unknown names.
- `expectile_loss`, `mc_q_loss`, and `iql_losses`: implement MC and IQL critic objectives.
- `CriticTrainingConfig`: loads strict JSON training configuration and rejects unknown keys.
- `save_critic_checkpoint` / `load_critic_checkpoint`: persist training state and validate compatibility manifests.
- `CriticTrainer`: trains frozen-backbone MC or IQL critics with single- or multi-process gradient synchronization.
- `wan_va/configs/critic_phase1.example.json`: provides an executable Phase 1 critic configuration template.
- Focused tests cover chunk transitions, masked pooling, MC/IQL targets, strict config validation, and checkpoint compatibility.
- `dataset_requirements.md`: defines the episode outcome fields required by RL dataset annotation.

### Changed

- `WanTransformer3DModel.forward_train`: optionally returns normalized action tokens `[B,F,N,D]` and video tokens `[B,F,S,D]`.
- `WanTransformer3DModel.forward`: forwards `return_features=True` without changing default outputs.
- `LatentLeRobotDataset.parse_meta`: propagates episode success and termination metadata into segment records.
- `LatentLeRobotDataset.get_rl_segment_metadata`: exposes segment frame counts and outcome metadata.
- `MultiLatentLeRobotDataset.get_rl_segment_metadata`: maps global sample indices to local RL metadata.
- `wan_va.__getattr__`: lazy-loads optional subpackages so isolated RL modules do not import unrelated dependencies.
- `DESIGN.md` / `IMPLEMENTATION.md`: define one RL action as the actions aligned to `infer_latent_chunk_size` latent frames.
- Inference guidance remains intentionally unimplemented.

### Transition Logic

```text
storage clip
  -> split valid latent frames into K-frame RL chunks
  -> link chunks chronologically within each episode
  -> mark only the final episode chunk terminal
  -> propagate sparse success rewards backward for MC returns
```

### Verification

- `PYTHONPATH=. pytest -q tests/test_rl_transitions.py tests/test_rl_critics.py tests/test_rl_config.py tests/test_rl_checkpoint.py`: 10 passed.
