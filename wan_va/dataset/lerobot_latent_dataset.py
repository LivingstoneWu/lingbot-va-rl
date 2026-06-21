# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

os.system('export WANDB_MODE=offline')

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory, followlinks=True):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    print('repo_list', len(repo_list))
    if not repo_list:
        raise RuntimeError(
            f"No datasets found under dataset_path='{config.dataset_path}'. "
            "Check that the path exists, contains symlinks or subdirectories with "
            "meta/info.json, and that symlinks are not broken."
        )
    ## 修改by lhc
    # with Pool(num_init_worker) as pool:
    #     datasets_out_lst = pool.map(construct_func, repo_list)
    for repo in repo_list:
        print("loading:", repo)
        datasets_out_lst.append(construct_func(repo))
    
    print("len dataset out list", len(datasets_out_lst))
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=1,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )
        print('finish')

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            print("dset_id", dset_id, self._datasets)
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

    def get_missing_jepa_files(self) -> list[str]:
        missing: list[str] = []
        for ds in self._datasets:
            missing.extend(ds.get_missing_jepa_files())
        return missing

    def get_rl_segment_metadata(self, idx: int) -> dict:
        dataset_id = self.item_id_to_dataset_id[idx]
        local_idx = idx - self.acc_dset_num[dataset_id]
        metadata = dict(
            self._datasets[dataset_id].get_rl_segment_metadata(local_idx)
        )
        metadata['dataset_id'] = dataset_id
        metadata['dataset_idx'] = idx
        return metadata

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.jepa_loss_enabled = getattr(config, 'jepa_loss_enabled', False)
        if self.jepa_loss_enabled:
            self.jepa_path = Path(repo_id) / 'jepa'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            outcome = {
                name: value[name]
                for name in (
                    "success",
                    "termination_frame",
                    "termination_reason",
                    "truncated",
                )
                if name in value
            }
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                    **outcome,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def get_rl_segment_metadata(self, idx: int) -> dict:
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_file = (
            Path(self.latent_path)
            / f"chunk-{episode_chunk:03d}"
            / self.used_video_keys[0]
            / (
                f"episode_{episode_index:06d}_"
                f"{cur_meta['start_frame']}_{cur_meta['end_frame']}.pth"
            )
        )
        latent_data = torch.load(latent_file, map_location="cpu", weights_only=False)
        metadata = dict(cur_meta)
        metadata["dataset_idx"] = idx
        metadata["latent_frame_count"] = int(latent_data["latent_num_frames"])
        return metadata

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file), f"Missing latent file: {latent_file}"
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    # def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
    #     act_shift = int(latent_frame_ids[0] - local_start_frame)
    #     frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
    #     action = action[act_shift:]
    #     if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
    #         left_action = get_relative_pose(action[:, :7])
    #         right_action = get_relative_pose(action[:, 8:15])
    #         action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
    #     action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

    #     latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
    #     required_action_num = latent_frame_num * frame_stride * 4

    #     action = action[:required_action_num]
    #     action_mask = np.ones_like(action, dtype='bool')
    #     assert action.shape[0] == required_action_num

    #     # COMMENT: incoming action: original shape, no padding; inverse_used_action_channel_ids: dim=30, each mapping to the index of the 
    #     # COMMENT: corresponding channel in the original action, not used channels points to action_dim + 1, here would be the additional padded 0 dimension.
    #     action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #     action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

    #     action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
    #     action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
    #     action_aligned = (action_aligned - self.q01) / (
    #             self.q99 - self.q01 + 1e-6) * 2. - 1.
    #     print("normalized action: ", action_aligned[0])
    #     print("action norm: ", torch.linalg.norm(action_aligned[0]))
    #     action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
    #     action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
    #     action_aligned *= action_mask_aligned
    #     return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    
    def _action_post_process(self, local_start_frame, latent_frame_ids, action, latent_frame_num):
        """
        Build the (C, F, N, 1) action tensor that aligns with the VAE latent frames.

        Alignment:
          latent frame f  ←→  action rows [f*N : (f+1)*N]
          where N = action_per_frame = frame_stride * 4.

        The HF fetch in __getitem__ already provides enough rows to cover all F*N
        slots, clamped to episode length.  If the episode ends before F*N rows are
        available (last chunk of an episode), the deficit is zero-padded here and
        the corresponding mask entries are set to False so those slots are excluded
        from the training loss.

        act_shift handles the rare case where action_config.start_frame differs
        from latent_frame_ids[0] (e.g. stale action_config from an older extraction
        run).  For freshly extracted data this is always 0.
        """
        act_shift = int(latent_frame_ids[0]) - local_start_frame
        frame_stride = int(latent_frame_ids[1] - latent_frame_ids[0])
        action = action[act_shift:]

        # Pad HF action to a common width before slicing so that datasets without
        # certain dimensions (e.g. mobile chassis/head) can share the same
        # action_slicing_ids as datasets that do have them.  Missing dimensions are
        # zero-padded, which trains the model to predict zero for those dims on
        # non-mobile tasks — the correct ground truth.
        # Set action_pad_to_dim to the widest HF action dim across all datasets.
        # Omit (or set to None) to skip padding (backward compatible).
        action_pad_to_dim = getattr(self.config, 'action_pad_to_dim', None)
        if action_pad_to_dim is not None and action.shape[1] < action_pad_to_dim:
            pad_width = action_pad_to_dim - action.shape[1]
            action = np.pad(action, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

        # Optional dim selection for datasets whose HF action column contains more
        # dimensions than the pipeline expects.  action_slicing_ids is a list of
        # integer column indices into the raw (T, D_hf) HF action tensor; the
        # selected columns must be ordered to match used_action_channel_ids.
        # Omit the key entirely for datasets that already have the right dims.
        # 从原数据中截取要用的动作维度
        action_slicing_ids = getattr(self.config, 'action_slicing_ids', None)
        if action_slicing_ids is not None:
            action = action[:, action_slicing_ids]   # (T, D_hf) → (T, n_used)

        if self.config.env_type == 'robotwin_tshape':
            left_action  = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate(
                [left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1
            )

        # Causal VAE 1,4,4,… compensation: latent frame 0 covers only 1 sampled input
        # frame (frame_stride real action rows), while every subsequent latent frame
        # covers 4 sampled input frames (frame_stride*4 rows).  Repeat each of the
        # first frame_stride rows 4× so that all latent frames get a uniform N-row
        # slot and the rearrange below stays simple.
        first_upsampled = np.repeat(action[:frame_stride], 4, axis=0)  # (frame_stride*4, C)
        action = np.concatenate([first_upsampled, action[frame_stride:]], axis=0)

        required_action_num = latent_frame_num * frame_stride * 4

        # Validity mask: True for real action rows, False for episode-end padding.
        # Created after upsampling so the repeated rows are also marked valid.
        available = action.shape[0]
        action_mask = np.ones_like(action, dtype='bool')

        # 万一动作数据异常，padding到足够的长度，并且mask掉padding的部分
        if available < required_action_num:
            # Last chunk of an episode: pad the deficit with zeros and mask them out.
            deficit = required_action_num - available
            action      = np.pad(action,      ((0, deficit), (0, 0)), mode='constant', constant_values=0)
            action_mask = np.pad(action_mask, ((0, deficit), (0, 0)), mode='constant', constant_values=False)
        else:
            action      = action[:required_action_num]
            action_mask = action_mask[:required_action_num]

        assert action.shape[0] == required_action_num, (
            f"action shape {action.shape[0]} != required {required_action_num}"
        )

        # 加一列0用于被inverse_used_action_channel_ids索引到，表示不使用的动作维度会被映射到这一列，后续归一化和训练时自然被0填充，不影响训练。
        action_paded      = np.pad(action,      ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned      = action_paded[:,       self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:,  self.config.inverse_used_action_channel_ids]

        denom = np.maximum(self.q99 - self.q01, 1e-2)
        action_aligned = np.where(
            action_mask_aligned,
            (action_aligned - self.q01) / (denom + 1e-6) * 2. - 1.,
            0.0,
        )

        action_aligned      = rearrange(action_aligned,      "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _load_jepa_target(
        self, episode_index: int, F_real: int,
        latent_h: int = 0, latent_w: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and temporally align JEPA features for one episode.

        Temporal alignment (single-chunk-per-episode assumed):
            VAE latent 0        → JEPA token 0           (prepended duplicate frame)
            VAE latent j (j≥1)  → mean(token 2j-1, token 2j)

        Spatially, per-camera features are concatenated to match _cat_video_latents,
        then 2×2 avg-pooled to match the DiT token grid (patch_size=(1,2,2)).

        latent_h / latent_w: combined spatial dims of the VAE latent for this
        episode (shape[2] / shape[3] of the [C,F,H,W] tensor).  Used to size
        the zero-fallback tensor correctly: H_tok = latent_h // 2, W_tok = latent_w // 2.

        Returns (jepa_target [F_real, H_tok, W_tok, D], jepa_available scalar bool).
        """
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        jepa_file = (
            self.jepa_path
            / f'chunk-{episode_chunk:03d}'
            / f'episode_{episode_index:06d}.pt'
        )

        def _zero(d=1664):
            # Zero fallback sized to the DiT token grid: latent_h//2 × latent_w//2.
            # latent_h/w are the combined (post-cat) VAE latent spatial dims passed
            # in from __getitem__, so this is always correct regardless of env_type.
            h_tok = latent_h // 2 if latent_h else 1
            w_tok = latent_w // 2 if latent_w else 1
            return torch.zeros(F_real, h_tok, w_tok, d, dtype=torch.bfloat16)

        if not jepa_file.exists():
            print(f'[JEPA] Missing: {jepa_file}')
            return _zero(), torch.tensor(False)

        try:
            jepa_data = torch.load(jepa_file, weights_only=False)

            aligned_list: list[torch.Tensor] = []
            for cam_key in self.used_video_keys:
                if cam_key not in jepa_data:
                    ref = next(
                        (jepa_data[k] for k in jepa_data if k != 'frame_ids'), None
                    )
                    H_p = ref.shape[1] if ref is not None else 16
                    W_p = ref.shape[2] if ref is not None else 16
                    D   = ref.shape[3] if ref is not None else 1664
                    aligned_list.append(
                        torch.zeros(F_real, H_p, W_p, D, dtype=torch.bfloat16)
                    )
                    continue

                feats = jepa_data[cam_key]       # [T_jepa, H', W', D]
                H_p, W_p, D = feats.shape[1], feats.shape[2], feats.shape[3]

                # Temporal alignment: F_real latent frames → 2*(F_real-1)+1 JEPA tokens
                # Latent 0      → JEPA token 0  (prepended duplicate frame)
                # Latent j≥1   → mean(token 2j-1, token 2j)
                #
                # Robustness: when T_jepa is even, feats[1::2] is one token longer
                # than feats[2::2] (last odd index has no paired even token).  We
                # take min(len(odd), len(even)) pairs and repeat the last valid
                # frame for any remaining latent slots instead of crashing.
                aligned = torch.empty(F_real, H_p, W_p, D, dtype=torch.float32)
                aligned[0] = feats[0].float()
                if F_real > 1:
                    tok_odd  = feats[1::2].float()   # all odd-indexed tokens
                    tok_even = feats[2::2].float()   # all even-indexed tokens
                    n_pairs  = min(len(tok_odd), len(tok_even), F_real - 1)
                    if n_pairs > 0:
                        aligned[1 : 1 + n_pairs] = (
                            tok_odd[:n_pairs] + tok_even[:n_pairs]
                        ) * 0.5
                    if n_pairs < F_real - 1:
                        # Repeat the last valid aligned frame for the shortfall
                        last = aligned[n_pairs] if n_pairs > 0 else aligned[0]
                        aligned[1 + n_pairs :] = last.unsqueeze(0).expand(
                            F_real - 1 - n_pairs, -1, -1, -1
                        )
                aligned_list.append(aligned.to(torch.bfloat16))

            # Spatial concat matching _cat_video_latents (concat along W', dim=2)
            if self.config.env_type == 'robotwin_tshape':
                wrist = torch.cat(aligned_list[1:], dim=2)   # [F, H', W'*2, D]
                jepa_target = torch.cat([wrist, aligned_list[0]], dim=1)
            else:
                jepa_target = torch.cat(aligned_list, dim=2)  # [F, H', W'*n, D]

            # JEPA patches are 16×16 px; the DiT further patchifies the VAE latent
            # by patch_size=(1,2,2), so each DiT token covers 32×32 px.  Pool the
            # JEPA tokens 2×2 spatially so jepa_target matches the DiT token grid.
            jepa_target = (
                jepa_target                              # [F, H', W', D]
                .permute(0, 3, 1, 2)                    # [F, D, H', W']
                .contiguous()
            )
            jepa_target = torch.nn.functional.avg_pool2d(
                jepa_target, kernel_size=2, stride=2
            )                                            # [F, D, H'//2, W'//2]
            jepa_target = jepa_target.permute(0, 2, 3, 1).to(torch.bfloat16)  # [F, H'//2, W'//2, D]

            return jepa_target, torch.tensor(True)

        except Exception as exc:
            print(f'[JEPA] Error loading {jepa_file}: {exc}')
            return _zero(), torch.tensor(False)

    def get_missing_jepa_files(self) -> list[str]:
        """Return paths of JEPA episode files that do not yet exist on disk."""
        if not self.jepa_loss_enabled:
            return []
        missing: list[str] = []
        seen: set[int] = set()
        for meta in self.new_metas:
            ep_idx = meta['episode_index']
            if ep_idx in seen:
                continue
            seen.add(ep_idx)
            ep_chunk = self.meta.get_episode_chunk(ep_idx)
            jepa_file = (
                self.jepa_path
                / f'chunk-{ep_chunk:03d}'
                / f'episode_{ep_idx:06d}.pt'
            )
            if not jepa_file.exists():
                missing.append(str(jepa_file))
        return missing

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index    = cur_meta["episode_index"]
        local_start_frame = cur_meta["start_frame"]
        local_end_frame   = cur_meta["end_frame"]

        ori_data_dict = self._get_range_latent_data(local_start_frame, local_end_frame, episode_index)

        # 对应的是被VAE编码之前，降采样后的帧id
        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]

        # frame_stride and latent_frame_num are derived from the latent file itself so
        # this stays correct even if action_config was generated by an older extraction run.
        frame_stride     = int(latent_frame_ids[1] - latent_frame_ids[0])
        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        act_shift        = int(latent_frame_ids[0]) - local_start_frame

        # Real rows to fetch before the 4× upsampling of the first latent frame's slot:
        # latent 0 needs frame_stride rows; each subsequent latent needs frame_stride*4.
        real_action_num = frame_stride + (latent_frame_num - 1) * frame_stride * 4

        # 动作取到当前chunk最后frame对应的位置
        ep_global_end    = int(self.episode_data_index["to"][episode_index])
        global_start     = self._get_global_idx(episode_index, local_start_frame)
        action_fetch_end = min(global_start + act_shift + real_action_num, ep_global_end)

        hf_data_frames = self._get_range_hf_data(global_start, action_fetch_end)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
            local_start_frame, latent_frame_ids, ori_data_dict['action'], latent_frame_num
        )

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)   # (C, F, H, W)
        F_real    = out_dict['latents'].shape[1]
        latent_h  = out_dict['latents'].shape[2]
        latent_w  = out_dict['latents'].shape[3]

        # ── JEPA target loading ───────────────────────────────────────────────
        if self.jepa_loss_enabled:
            out_dict['jepa_target'], out_dict['jepa_available'] = (
                self._load_jepa_target(episode_index, F_real, latent_h, latent_w)
            )

        # ── Pad F dimension to max_latent_frames ──────────────────────────────
        # Chunks vary in latent frame count (shorter at episode boundaries).
        # Padding to a fixed max lets the default collate stack a batch of B > 1.
        # MYEDIT: noticed that padding with batch_size = 1 causes significant delays, only pad when batch_size > 1
        if self.config.batch_size == 1:
            F_max = F_real
        else: 
            F_max  = self.config.max_latent_frames
        F_pad  = F_max - F_real
        assert F_pad >= 0, (
            f"Chunk has {F_real} latent frames but config.max_latent_frames={F_max}. "
            "Increase max_latent_frames."
        )
        if F_pad > 0:
            # torch.nn.functional.pad specifies padding from the last dim backwards.
            # Both (C,F,H,W) and (C,F,N,1) are 4-D with F at dim-1, so the same
            # tuple pads dim-1 (F) on the right: (W/1: 0, H/N: 0, F: F_pad).
            f_pad = (0, 0, 0, 0, 0, F_pad)
            out_dict['latents']      = torch.nn.functional.pad(out_dict['latents'], f_pad)
            out_dict['actions']      = torch.nn.functional.pad(out_dict['actions'], f_pad)
            # actions_mask is bool; pad with 0 → False (masked out)
            out_dict['actions_mask'] = torch.nn.functional.pad(
                out_dict['actions_mask'].float(), f_pad
            ).bool()
            # jepa_target [F, H', W', D]: F is dim-0, so pad tuple has 4 pairs.
            if self.jepa_loss_enabled:
                jepa_f_pad = (0, 0, 0, 0, 0, 0, 0, F_pad)
                out_dict['jepa_target'] = torch.nn.functional.pad(
                    out_dict['jepa_target'], jepa_f_pad
                )

        # Frame-level validity mask: True = real latent frame, False = padding.
        # Shape (F_max,); after collate becomes (B, F_max).
        latents_mask = torch.zeros(F_max, dtype=torch.bool)
        latents_mask[:F_real] = True
        out_dict['latents_mask'] = latents_mask

        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            #num_workers=32,
            num_workers=0,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
