import os

os.environ['HF_LEROBOT_HOME'] = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge'

import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


HF_LEROBOT_HOME = Path("/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/ur5e")

# Example:
# /liujinxin/code/lhc/wy/wms/lingbot-va/datasets_ori/ur5e/stack_color_blocks/2026042_stack_blocks
RAW_BASE_DIR = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets_ori/ur5e/stack_color_blocks"

ACTION_KEY = "joint_angles"
ACTION_DIM = 7
CHECK_DIMS = 6


@dataclass
class TakeDataFiles:
    take_num: str
    take_dir: str
    parent_dir: str
    data_json: Optional[str] = None

    def is_complete(self) -> bool:
        return self.data_json is not None and os.path.exists(self.data_json)


def process_take_directory(take_dir: str, parent_dir: str) -> Optional[TakeDataFiles]:
    take_name = os.path.basename(take_dir)
    if not take_name.startswith("take"):
        return None

    take_num = take_name.replace("take", "", 1)
    take = TakeDataFiles(
        take_num=take_num,
        take_dir=take_dir,
        parent_dir=parent_dir,
    )
    data_json_path = os.path.join(take_dir, "data.json")
    if os.path.exists(data_json_path):
        take.data_json = data_json_path
    return take


def find_all_take_files(base_dir: str) -> List[TakeDataFiles]:
    """
    递归查找base_dir下所有take目录的数据文件和相关资源
    支持形如: base_dir/*/take* 或 base_dir/take* 的目录结构

    Args:
        base_dir: 基础目录

    Returns:
        TakeDataFiles对象列表
    """
    if not os.path.exists(base_dir):
        print(f"警告：目录 '{base_dir}' 不存在")
        return []

    all_takes = []

    # 方法1: 直接在base_dir下查找take*目录
    direct_takes = glob.glob(os.path.join(base_dir, "take*"))
    for take_dir in direct_takes:
        if os.path.isdir(take_dir):
            take = process_take_directory(take_dir, base_dir)
            if take and take.is_complete():
                all_takes.append(take)

    # 方法2: 在base_dir的子目录中查找take*目录
    sub_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    for sub_dir in sub_dirs:
        # 跳过已经处理过的take目录
        if os.path.basename(sub_dir).startswith("take"):
            continue

        take_dirs = glob.glob(os.path.join(sub_dir, "take*"))
        for take_dir in take_dirs:
            if os.path.isdir(take_dir):
                take = process_take_directory(take_dir, sub_dir)
                if take and take.is_complete():
                    all_takes.append(take)

    # 按take编号排序（非数字按0处理）
    all_takes.sort(key=lambda x: int(x.take_num) if x.take_num.isdigit() else 0)

    print(f"在 {base_dir} 中共找到 {len(all_takes)} 个有效take目录")
    return all_takes


def extract_take_actions(data_json_path: str) -> np.ndarray:
    """
    读取一个take的 data.json，并提取 ACTION_KEY 为 (N, 7)。
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{data_json_path}: expected list, got {type(data)}")

    out: List[np.ndarray] = []
    for i, step in enumerate(data):
        if not isinstance(step, dict):
            raise ValueError(f"{data_json_path}: step {i} is not dict")
        if ACTION_KEY not in step:
            raise KeyError(f"{data_json_path}: missing key '{ACTION_KEY}' at step {i}")

        arr = np.asarray(step[ACTION_KEY], dtype=np.float32).reshape(-1)
        if arr.shape[0] < ACTION_DIM:
            raise ValueError(
                f"{data_json_path}: step {i} has {arr.shape[0]} dims, expected >= {ACTION_DIM}"
            )
        if arr.shape[0] > ACTION_DIM:
            arr = arr[:ACTION_DIM]
        out.append(arr)

    return np.stack(out, axis=0) if out else np.zeros((0, ACTION_DIM), dtype=np.float32)


def build_episode_action_map(
    parquet_files: List[Path],
    take_files: List[TakeDataFiles],
) -> Dict[int, np.ndarray]:
    """
    按照 take 排序顺序与 episode_index 排序顺序一一对应，构建:
      episode_index -> source_actions(N,7)
    """
    meta = pd.concat(
        [pd.read_parquet(pf, columns=["episode_index", "frame_index"]) for pf in parquet_files],
        ignore_index=True,
    )
    episode_counts: Dict[int, int] = meta.groupby("episode_index").size().to_dict()
    episode_indices = sorted(episode_counts.keys())

    if len(episode_indices) != len(take_files):
        raise RuntimeError(
            "Episode count mismatch: "
            f"converted={len(episode_indices)}, takes={len(take_files)}"
        )

    episode_action_map: Dict[int, np.ndarray] = {}
    for ep_idx, take in zip(episode_indices, take_files):
        src_actions = extract_take_actions(take.data_json)  # (N,7)
        n_conv = int(episode_counts[ep_idx])
        n_src = int(src_actions.shape[0])
        if n_src != n_conv:
            raise RuntimeError(
                f"Frame count mismatch for episode_index={ep_idx} <-> {take.take_dir}: "
                f"source={n_src}, converted={n_conv}"
            )
        episode_action_map[ep_idx] = src_actions
    return episode_action_map


def patch_parquet_actions(
    parquet_path: Path,
    episode_action_map: Dict[int, np.ndarray],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tuple[int, int]:
    """
    按 row 的 (episode_index, frame_index) 对齐 source action。
    安全检查:
      converted action[:, :6] 与 source action[:, :6] 必须 allclose。
    修补:
      converted action[:, 6] <- source action[:, 6]

    Returns:
      (rows_patched, rows_checked)
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    if "action" not in df.columns:
        raise KeyError(f"{parquet_path}: missing column 'action'")
    if "episode_index" not in df.columns or "frame_index" not in df.columns:
        raise KeyError(f"{parquet_path}: missing episode_index/frame_index")

    action_full = np.stack(df["action"].values).astype(np.float32)
    if action_full.shape[1] < ACTION_DIM:
        raise ValueError(
            f"{parquet_path}: action dim={action_full.shape[1]}, expected >= {ACTION_DIM}"
        )

    ep_arr = df["episode_index"].to_numpy(dtype=np.int64)
    frame_arr = df["frame_index"].to_numpy(dtype=np.int64)

    rows_patched = 0
    rows_checked = 0
    for i in range(len(df)):
        ep_idx = int(ep_arr[i])
        if ep_idx not in episode_action_map:
            continue
        fi = int(frame_arr[i])
        src_actions = episode_action_map[ep_idx]
        if fi < 0 or fi >= src_actions.shape[0]:
            raise RuntimeError(
                f"{parquet_path}: frame_index out of range at row={i}, "
                f"episode_index={ep_idx}, frame_index={fi}, source_len={src_actions.shape[0]}"
            )

        src = src_actions[fi]
        rows_checked += 1

        # Security check: first 6 dims should already match.
        if not np.allclose(action_full[i, :CHECK_DIMS], src[:CHECK_DIMS], atol=atol, rtol=rtol):
            max_abs = float(np.max(np.abs(action_full[i, :CHECK_DIMS] - src[:CHECK_DIMS])))
            raise RuntimeError(
                f"{parquet_path}: safety check failed at row={i}, episode_index={ep_idx}, frame_index={fi}. "
                f"first-{CHECK_DIMS}-dim mismatch (max_abs={max_abs:.6g})"
            )

        action_full[i, ACTION_DIM - 1] = src[ACTION_DIM - 1]
        rows_patched += 1

    df["action"] = list(action_full)
    patched_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(patched_table, parquet_path)
    return rows_patched, rows_checked


def main(
    repo_name: str,
    raw_base_dir: str = RAW_BASE_DIR,
    dry_run: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """
    使用原始 take*/data.json 中的 joint_angles 修补已转换数据集的 action 最后一维。

    映射策略（关键）:
      - take 文件顺序：find_all_take_files(base_dir) 的排序顺序
      - converted episode 顺序：episode_index 升序
      - 二者按顺序一一对应

    只修补:
      action[:, 6]

    安全检查:
      action[:, 0:6] 必须和源数据完全对齐（allclose），否则中止。
    """
    dataset_path = HF_LEROBOT_HOME / repo_name
    parquet_files = sorted((dataset_path / "data").glob("**/*.parquet"))
    if not parquet_files:
        print(f"ERROR: no parquet files found under {dataset_path / 'data'}")
        return

    take_files = find_all_take_files(raw_base_dir)
    if not take_files:
        print("ERROR: no valid take data.json files found")
        return

    print(f"Converted dataset : {dataset_path}")
    print(f"Raw base dir      : {raw_base_dir}")
    print(f"Parquet files     : {len(parquet_files)}")
    print(f"Take files        : {len(take_files)}")
    print(f"Dry run           : {dry_run}")

    episode_action_map = build_episode_action_map(parquet_files, take_files)
    print(f"Episode mapping   : {len(episode_action_map)} episodes")

    if dry_run:
        print("Dry run complete. Mapping and frame counts verified; nothing written.")
        return

    total_patched = 0
    total_checked = 0
    for pf in parquet_files:
        rows_patched, rows_checked = patch_parquet_actions(
            pf,
            episode_action_map=episode_action_map,
            atol=atol,
            rtol=rtol,
        )
        if rows_checked > 0:
            print(f"{pf.name}: patched {rows_patched} rows (checked {rows_checked})")
        total_patched += rows_patched
        total_checked += rows_checked

    print(f"Done. Patched {total_patched} rows, checked {total_checked} rows.")


if __name__ == "__main__":
    tyro.cli(main)
