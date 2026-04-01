import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

DATASET_ROOT = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge"


def detect_camera_key(latents_dir: Path) -> Optional[str]:
    """
    从latents目录结构中自动检测第一个可用的摄像头键名。
    结构: latents/chunk-*/observation.images.xxx/
    """
    for chunk_dir in sorted(latents_dir.glob('chunk-*')):
        if not chunk_dir.is_dir():
            continue
        for cam_dir in sorted(chunk_dir.iterdir()):
            if cam_dir.is_dir():
                return cam_dir.name
    return None


def get_episode_windows(latents_dir: Path, episode_id: int, camera_key: str) -> List[Tuple[int, int]]:
    """
    通过扫描latent文件名，获取指定episode的所有时间窗口。

    文件名格式: episode_{id:06d}_{start_frame}_{end_frame}.pth

    Returns:
        按start_frame排序的 (start_frame, end_frame) 列表
    """
    windows = []
    pattern = f'chunk-*/{camera_key}/episode_{episode_id:06d}_*.pth'
    for pth_file in latents_dir.glob(pattern):
        parts = pth_file.stem.split('_')
        # stem例: episode_000000_0_994 -> parts[-2]=0, parts[-1]=994
        try:
            start = int(parts[-2])
            end = int(parts[-1])
            windows.append((start, end))
        except (ValueError, IndexError):
            print(f"  警告: 无法解析文件名 {pth_file.name}，跳过")
    return sorted(windows)


def update_episodes_jsonl(dataset_path: Path):
    """
    读取episodes.jsonl，根据实际latent文件名更新每个episode的action_config，
    并将结果写回episodes.jsonl（备份原文件到episodes_original.jsonl）。
    """
    meta_dir = dataset_path / 'meta'
    latents_dir = dataset_path / 'latents'
    episodes_file = meta_dir / 'episodes.jsonl'
    backup_file = meta_dir / 'episodes_original.jsonl'

    if not episodes_file.exists():
        print(f"错误: {episodes_file} 不存在")
        return

    if not latents_dir.exists():
        print(f"错误: latents目录 {latents_dir} 不存在，请先运行latent提取")
        return

    # 自动检测摄像头键名
    camera_key = detect_camera_key(latents_dir)
    if camera_key is None:
        print(f"错误: 在 {latents_dir} 下未找到任何latent文件")
        return
    print(f"检测到摄像头键名: {camera_key}")

    # 备份原始文件（只在首次运行时备份）
    if not backup_file.exists():
        shutil.copy2(episodes_file, backup_file)
        print(f"原始文件已备份到: {backup_file}")
    else:
        print(f"备份文件已存在，跳过备份: {backup_file}")

    # 读取并更新每个episode
    episodes = []
    missing_latents = []

    with open(episodes_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            episode_id = ep['episode_index']

            windows = get_episode_windows(latents_dir, episode_id, camera_key)

            action_text = ''
            tasks = ep.get('tasks', [])
            if tasks:
                action_text = tasks[0] if isinstance(tasks, list) else tasks

            ep['action_config'] = [
                {
                    'start_frame': start,
                    'end_frame': end,
                    'action_text': action_text,
                    'skill': '',
                }
                for start, end in windows
            ]

            if not windows:
                missing_latents.append(episode_id)

            episodes.append(ep)

    # 写回episodes.jsonl
    with open(episodes_file, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + '\n')

    print(f"\n更新完成: {len(episodes)} 个episode写入 {episodes_file}")
    total_windows = sum(len(ep['action_config']) for ep in episodes)
    print(f"共生成 {total_windows} 个action_config窗口")

    if missing_latents:
        print(f"\n警告: 以下 {len(missing_latents)} 个episode未找到latent文件（action_config为空）:")
        print(f"  {missing_latents[:20]}{'...' if len(missing_latents) > 20 else ''}")

    # 显示示例
    if episodes:
        print(f"\n第一个episode示例:")
        print(json.dumps(episodes[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="根据实际latent文件名更新episodes.jsonl中的action_config字段"
    )
    parser.add_argument(
        '--repo-name',
        type=str,
        required=True,
        help="数据集名称（DATASET_ROOT下的子目录名）"
    )
    args = parser.parse_args()

    dataset_path = Path(DATASET_ROOT) / args.repo_name
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return

    update_episodes_jsonl(dataset_path)


if __name__ == '__main__':
    main()
