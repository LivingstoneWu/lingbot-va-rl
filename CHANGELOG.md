# Changelog

## 2026-06-21 - Critic Training Workflow

### Added

- `wan_va_server_q_guiding.py`: adds a separate VA server entry point that preserves the existing server interface while applying Q-gradient guidance in the action denoising loop.
- `QGuidedVA_Server`: loads a critic checkpoint, validates chunk/action feature compatibility, and evaluates Q on clean action estimates during action sampling.
- `QGuidanceAdapter` / `TwinMLPQGuidanceAdapter`: define the server-facing Q objective contract and current twin-Q implementation.
- `guidance.py`: adds denoising-time scaling, clean feature-input construction, action guidance masks, Q checkpoint artifact loading, and a registry-backed guidance adapter.
- `tests/test_rl_guidance.py`: covers denoising-time scaling, invalid/clamped action masking, Q artifact loading, and manifest mismatch rejection.
- `workflow.md`: documents dataset preflight, base/critic config ownership, per-rank loader settings, attention-window semantics, launch commands, monitoring, and checkpoint resume behavior.
- `extract_latent_vae_robotwin.py`: validates RobotWin camera ordering and maps the high camera to full resolution and wrist cameras to half resolution before VAE encoding.
- `test_robotwin_latent_extraction.py`: covers valid T-shape sizes and rejects incompatible camera order, resolution, and environment configs.

### Changed

- `DESIGN.md`: uses `$...$` for inline equations and `$$...$$` for display equations.
- `extract_latent_vae.main`: accepts optional config-validation and per-camera-size hooks while retaining uniform sizing by default.
- `extract_robotwin.sh`: launches the dedicated RobotWin extractor with the accessible RL config and `256x320` high-camera resolution.
- `_prepare_clean_input`: casts floating dataset tensors to `base_config.param_dtype` before frozen-transformer feature extraction.
- `IMPLEMENTATION.md`: documents the BF16-transformer/FP32-critic dtype path and additional changes required for full FP32 backbone support.
- `CriticTrainer._parameter_norm`: computes the global L2 norm of trainable critic parameters for logging.
- `CriticTrainer.train`: writes `total_loss`, `q_loss`, unweighted `value_loss`, gradient norm, and parameter norm to `loss.jsonl` without a legacy `loss` alias.
- `CriticTrainingConfig.__post_init__`: rejects non-positive `log_interval` values.
- `workflow.md`: documents the component-loss fields and MC value-loss behavior in `loss.jsonl`.
- `CriticTrainingConfig`: adds one-element `feature_layers`, `feature_aggregation`, selected-layer access, and normalization semantics.
- `WanTransformer3DModel.forward_train`: optionally returns final normalized or one raw post-block action/video feature stream without changing diffusion outputs.
- `WanTransformer3DModel.forward`: forwards the optional `critic_feature_layer` argument only through the training feature path.
- `CriticTrainer`: validates the selected block, uses it for current/predecessor extraction, and enforces feature-spec compatibility on resume.
- `save_critic_checkpoint`: persists layer, aggregation, and normalization in checkpoint schema version 3.
- `critic_phase1.example.json`: exposes `feature_layers=[-1]` and `feature_aggregation="single"` defaults.
- `wan_va.rl.__init__`: exports the Q-guidance artifact loader and denoising/mask helpers.
- `load_q_guidance_artifact`: accepts legacy schema-2 critic checkpoints only as final-normalized `feature_layers=[-1]` guidance artifacts.
- `QGuidedVA_Server._q_feature_extraction_context`: temporarily disables live KV cache use and restores FlexAttention masks during Q feature extraction to keep critic features current-chunk-only.
- `DESIGN.md` / `IMPLEMENTATION.md` / `workflow.md`: define feature taps, Phase 1 restrictions, future mixing, inference ownership, and first-version QGF guidance.

### Verification

- `PYTHONPATH=. pytest -q tests/test_rl_transitions.py tests/test_rl_critics.py tests/test_rl_config.py tests/test_rl_checkpoint.py`: 16 passed.
- `PYTHONPATH=. pytest -q tests/test_rl_guidance.py tests/test_rl_checkpoint.py tests/test_rl_critics.py tests/test_rl_config.py`: 16 passed.
- `python -m py_compile wan_va/rl/guidance.py wan_va/wan_va_server_q_guiding.py tests/test_rl_guidance.py`: passed.
- `PYTHONPATH=. pytest -q tests/test_rl_critics.py tests/test_rl_config.py`: 10 passed after extending component-loss logging.
- `git -C /luhongchao/wy/lingbot-va-rl diff --check`: passed after removing the legacy `loss` JSONL alias.
- Direct LingBot-environment assertion: `CriticTrainer._parameter_norm` returned `5.0` for trainable parameters `[3,4]` while excluding a frozen parameter.


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
