import os
import csv
import json
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import tyro
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置常量
HF_LEROBOT_HOME = Path("/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge")  # 请修改为实际路径
RAW_DATASET_NAMES = ["set_the_plates"]  # 请修改为实际数据集名称
PUSH_TO_HUB = False
# NOTE: keep NUM_WORKERS small (4–8). Each worker loads full video frames into
# memory and returns large numpy arrays through the IPC pipe. 80 workers ×
# ~300MB/episode = ~24GB peak — this causes silent OOM kills on the cluster.
NUM_WORKERS = 4
DATA_DIR = "/liujinxin/dataset/robochallenge/"

# 摄像头CSV文件默认路径（相对于脚本所在目录）
DEFAULT_CAMERAS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras.csv")


def get_single_arm_robot_types(cameras_csv_path: str) -> List[str]:
    """
    从cameras.csv读取所有单臂机器人类型（即没有wrist_camera_right的机器人）

    Args:
        cameras_csv_path: cameras.csv文件路径

    Returns:
        单臂机器人类型名称列表（保持CSV中的原始大小写）
    """
    types = []
    try:
        with open(cameras_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row['wrist_camera_right'].strip():
                    types.append(row['robot_type'].strip())
    except Exception as e:
        print(f"警告：无法读取cameras.csv {cameras_csv_path}: {e}")
    return types


def detect_robot_type_from_task_info(task_info: Dict, known_robot_types: List[str]) -> Optional[str]:
    """
    从task_info.json的task_desc.task_tag中检测机器人类型

    task_tag列表中恰好有一个tag与已知单臂机器人类型匹配。

    Args:
        task_info: task_info.json解析后的内容
        known_robot_types: 已知单臂机器人类型列表（来自cameras.csv）

    Returns:
        匹配到的机器人类型字符串（原始大小写），未找到则返回None
    """
    try:
        tags = task_info.get('task_desc', {}).get('task_tag', [])
        for tag in tags:
            for robot_type in known_robot_types:
                if tag.strip().lower() == robot_type.lower():
                    return robot_type  # 保持CSV中的原始大小写
    except Exception as e:
        print(f"警告：无法从task_info检测机器人类型: {e}")
    return None


def load_camera_config(cameras_csv_path: str, robot_type: str) -> Dict[str, Optional[str]]:
    """
    从cameras.csv加载指定机器人类型的摄像头配置

    Args:
        cameras_csv_path: cameras.csv文件路径
        robot_type: 机器人类型（如 ARX5, UR5, Franka）

    Returns:
        包含摄像头名称的字典: {'main': ..., 'wrist': ..., 'scene': ...}
    """
    try:
        with open(cameras_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['robot_type'].strip().lower() == robot_type.strip().lower():
                    return {
                        'main': row['main_camera'].strip() or None,
                        'wrist': row['wrist_camera'].strip() or None,
                        'scene': row['scene_camera'].strip() or None,
                    }
    except Exception as e:
        print(f"错误：无法读取cameras.csv {cameras_csv_path}: {e}")

    raise ValueError(f"在 {cameras_csv_path} 中未找到机器人类型: {robot_type}")


class EpisodeStateFiles:
    """表示一个episode的状态文件和相关资源（单臂）"""
    def __init__(self, episode_num: str, episode_dir: str, dataset_name: str = "",
                 global_task_info: Optional[str] = None, global_video_prompt: Optional[str] = None):
        self.episode_num = episode_num
        self.episode_dir = episode_dir
        self.dataset_name = dataset_name
        self.states_file: Optional[str] = None
        self.states_dir: Optional[str] = None
        self.videos_dir: Optional[str] = None

        # 特定视频文件（键名对应摄像头类型）
        self.main_camera_video: Optional[str] = None    # main_camera (→ top)
        self.wrist_camera_video: Optional[str] = None   # wrist_camera (→ wrist)
        self.scene_camera_video: Optional[str] = None   # scene_camera (→ scene)

        # 全局元数据文件（所有episode共享）
        self.global_task_info_file = global_task_info
        self.global_video_prompt_file = global_video_prompt

    def is_complete(self) -> bool:
        """检查是否包含状态文件"""
        return self.states_file is not None

    def has_main_cameras(self) -> bool:
        """检查是否包含主摄像头和腕部摄像头"""
        return self.main_camera_video is not None and self.wrist_camera_video is not None

    def has_task_info(self) -> bool:
        """检查是否有全局task_info.json"""
        return self.global_task_info_file is not None and os.path.exists(self.global_task_info_file)

    def has_video_prompt(self) -> bool:
        """检查是否有全局video_prompt.txt"""
        return self.global_video_prompt_file is not None and os.path.exists(self.global_video_prompt_file)

    def get_video_dict(self) -> Dict[str, Optional[str]]:
        """获取所有视频文件的字典"""
        return {
            'main': self.main_camera_video,
            'wrist': self.wrist_camera_video,
            'scene': self.scene_camera_video,
        }

    def get_files_dict(self) -> Dict[str, Any]:
        """获取所有文件的字典"""
        return {
            'states': self.states_file,
            'videos': self.get_video_dict(),
            'global_task_info': self.global_task_info_file,
            'global_video_prompt': self.global_video_prompt_file,
        }

    def get_file_paths(self) -> List[str]:
        """获取所有存在的文件路径列表"""
        paths = []
        if self.states_file:
            paths.append(self.states_file)
        if self.main_camera_video:
            paths.append(self.main_camera_video)
        if self.wrist_camera_video:
            paths.append(self.wrist_camera_video)
        if self.scene_camera_video:
            paths.append(self.scene_camera_video)
        if self.global_task_info_file:
            paths.append(self.global_task_info_file)
        if self.global_video_prompt_file:
            paths.append(self.global_video_prompt_file)
        return paths

    def __repr__(self) -> str:
        status = "完整" if self.is_complete() else "不完整"
        video_count = sum(1 for v in [self.main_camera_video, self.wrist_camera_video, self.scene_camera_video] if v)
        return f"EpisodeStateFiles(episode={self.episode_num}, status={status}, videos={video_count}/3)"


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    读取jsonl文件内容

    Args:
        file_path: jsonl文件路径

    Returns:
        解析后的JSON对象列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告：{file_path} 第 {line_num} 行JSON解析错误: {e}")
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}: {e}")

    return data


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    读取JSON文件内容

    Args:
        file_path: JSON文件路径

    Returns:
        解析后的JSON对象，如果失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误：无法读取JSON文件 {file_path}: {e}")
        return None


def read_text_file(file_path: str) -> Optional[str]:
    """
    读取文本文件内容

    Args:
        file_path: 文本文件路径

    Returns:
        文件内容字符串，如果失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"错误：无法读取文本文件 {file_path}: {e}")
        return None


def read_video_info(video_path: Optional[str]) -> Tuple[Optional[int], Optional[Tuple[int, int, int]]]:
    """
    读取视频文件的基本信息（不加载帧）

    Args:
        video_path: 视频文件路径

    Returns:
        (帧数, (高度, 宽度, 通道数))
    """
    if video_path is None or not os.path.exists(video_path):
        return None, None

    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        channels = 3  # RGB
        cap.release()
        return frame_count, (height, width, channels)
    except Exception as e:
        print(f"  警告：读取视频信息 {video_path} 时出错: {e}")
        return None, None


def load_global_metadata(base_dir: str) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str]]:
    """
    加载全局元数据文件

    Args:
        base_dir: 基础目录

    Returns:
        (task_info_path, video_prompt_path, task_info_content, video_prompt_content)
    """
    meta_dir = os.path.join(base_dir, "meta")
    task_info_path = os.path.join(meta_dir, "task_info.json")
    video_prompt_path = os.path.join(base_dir, "video_prompt.txt")

    task_info_content = None
    video_prompt_content = None

    if os.path.exists(task_info_path):
        task_info_content = read_json_file(task_info_path)
        if task_info_content:
            print(f"✓ 加载全局 task_info.json: {task_info_path}")
    else:
        print(f"⚠ 全局 task_info.json 不存在: {task_info_path}")
        task_info_path = None

    if os.path.exists(video_prompt_path):
        video_prompt_content = read_text_file(video_prompt_path)
        if video_prompt_content:
            print(f"✓ 加载全局 video_prompt.txt: {video_prompt_path}")
    else:
        print(f"⚠ 全局 video_prompt.txt 不存在: {video_prompt_path}")
        video_prompt_path = None

    return task_info_path, video_prompt_path, task_info_content, video_prompt_content


def find_all_episode_state_files(base_dir: str, camera_config: Dict[str, Optional[str]]) -> List[EpisodeStateFiles]:
    """
    查找所有episode的状态文件和相关资源，以episode为单位组织

    Args:
        base_dir: 基础目录名
        camera_config: 摄像头配置字典 {'main': ..., 'wrist': ..., 'scene': ...}

    Returns:
        EpisodeStateFiles对象列表，每个对象包含一个episode的所有文件信息
    """
    episode_files_map = {}
    dataset_name = os.path.basename(base_dir)

    if not os.path.exists(base_dir):
        print(f"警告：目录 '{base_dir}' 不存在")
        return []

    global_task_info_path, global_video_prompt_path, _, _ = load_global_metadata(base_dir)

    episode_pattern = os.path.join(base_dir, "data", "episode_*")
    episode_dirs = glob.glob(episode_pattern)

    print(f"在 {dataset_name} 中找到 {len(episode_dirs)} 个episode目录")

    for episode_dir in sorted(episode_dirs):
        episode_num = os.path.basename(episode_dir).replace("episode_", "")

        episode = EpisodeStateFiles(
            episode_num,
            episode_dir,
            dataset_name,
            global_task_info=global_task_info_path,
            global_video_prompt=global_video_prompt_path,
        )

        episode.states_dir = os.path.join(episode_dir, "states")
        episode.videos_dir = os.path.join(episode_dir, "videos")

        # 1. 检查states目录和文件
        if os.path.exists(episode.states_dir):
            states_path = os.path.join(episode.states_dir, "states.jsonl")
            if os.path.exists(states_path):
                episode.states_file = states_path

        # 2. 根据摄像头配置检查视频文件
        if os.path.exists(episode.videos_dir):
            if camera_config['main']:
                path = os.path.join(episode.videos_dir, f"{camera_config['main']}.mp4")
                if os.path.exists(path):
                    episode.main_camera_video = path

            if camera_config['wrist']:
                path = os.path.join(episode.videos_dir, f"{camera_config['wrist']}.mp4")
                if os.path.exists(path):
                    episode.wrist_camera_video = path

            if camera_config['scene']:
                path = os.path.join(episode.videos_dir, f"{camera_config['scene']}.mp4")
                if os.path.exists(path):
                    episode.scene_camera_video = path

        episode_files_map[episode_num] = episode

    return list(episode_files_map.values())


def shuffle_episode_state_files(episode_files: List[EpisodeStateFiles], seed: int = 42) -> List[EpisodeStateFiles]:
    """
    对episode列表进行shuffle

    Args:
        episode_files: EpisodeStateFiles对象列表
        seed: 随机种子，确保可重复性

    Returns:
        shuffle后的列表
    """
    random.seed(seed)
    shuffled = episode_files.copy()
    random.shuffle(shuffled)
    return shuffled


def print_episode_info(episodes: List[EpisodeStateFiles], title: str = "Episode列表"):
    """
    打印episode信息

    Args:
        episodes: EpisodeStateFiles对象列表
        title: 标题
    """
    print(f"\n{title} (共 {len(episodes)} 个episode):")
    print("-" * 100)
    print(f"{'序号':<4} {'Dataset':<15} {'Episode':<10} {'状态':<8} {'视频文件':<30}")
    print("-" * 100)

    for i, episode in enumerate(episodes, 1):
        status = "✓完整" if episode.is_complete() else "✗不完整"

        video_files = []
        if episode.main_camera_video:
            video_files.append("M")
        if episode.wrist_camera_video:
            video_files.append("W")
        if episode.scene_camera_video:
            video_files.append("S")
        video_status = f"{len(video_files)}/3 ({','.join(video_files)})" if video_files else "无"

        dataset_short = episode.dataset_name[:15] if len(episode.dataset_name) > 15 else episode.dataset_name

        print(f"{i:<4} {dataset_short:<15} episode_{episode.episode_num:<6} {status:<8} {video_status:<30}")

    print("-" * 100)


def process_episode_fast(episode: EpisodeStateFiles, global_task_info: Optional[Dict] = None,
                         global_video_prompt: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    快速处理单个episode（单臂），优化速度

    Args:
        episode: EpisodeStateFiles对象
        global_task_info: 全局task_info.json内容
        global_video_prompt: 全局video_prompt.txt内容

    Returns:
        (episode_data, tasks, video_prompt) 元组
    """
    try:
        # 1. 加载状态文件
        states_data = []
        if episode.states_file and os.path.exists(episode.states_file):
            states_data = read_jsonl_file(episode.states_file)

        # 2. 获取视频信息（不加载帧）
        read_video_info(episode.main_camera_video)
        read_video_info(episode.wrist_camera_video)
        read_video_info(episode.scene_camera_video)

        # 3. 从全局元数据获取任务信息
        tasks = ""
        if global_task_info:
            if isinstance(global_task_info, list) and len(global_task_info) > 0:
                if 'task_desc' in global_task_info[0] and 'prompt' in global_task_info[0]['task_desc']:
                    tasks = global_task_info[0]['task_desc']['prompt']
                elif 'task' in global_task_info[0]:
                    tasks = global_task_info[0]['task']
            elif isinstance(global_task_info, dict):
                if 'task_desc' in global_task_info and 'prompt' in global_task_info['task_desc']:
                    tasks = global_task_info['task_desc']['prompt']
                elif 'task' in global_task_info:
                    tasks = global_task_info['task']

        # 4. 从全局获取视频提示
        video_prompt = global_video_prompt if global_video_prompt else ""

        # 5. 生成episode数据（不包含视频帧）
        states_len = len(states_data)

        if states_len < 2:
            print(f"  警告: Episode {episode.episode_num} 数据长度不足")
            return ([], tasks, video_prompt)

        valid_len = states_len - 1
        episode_data = []

        for index in range(valid_len):
            eef_state = np.array(states_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
            eef_action = np.array(states_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
            joints_state = np.array(states_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
            joints_action = np.array(states_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
            gripper_state = float(states_data[index].get('gripper', 0.0))
            gripper_action = float(states_data[index + 1].get('gripper', 0.0))

            # state: eef(7) + joints(6) + gripper(1) = 14
            state = np.concatenate((
                eef_state,
                joints_state,
                np.array([gripper_state], dtype=np.float32),
            ))

            # action: eef(7) + joints(6) + gripper(1) = 14
            action = np.concatenate((
                eef_action,
                joints_action,
                np.array([gripper_action], dtype=np.float32),
            ))

            episode_data.append({
                'action': action,
                'state': state,
            })

        return (episode_data, tasks, video_prompt)

    except Exception as e:
        print(f"错误处理 episode {episode.episode_num}: {e}")
        import traceback
        traceback.print_exc()
        return ([], "", "")


def process_episode_with_frames(episode: EpisodeStateFiles, global_task_info: Optional[Dict] = None,
                                global_video_prompt: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    处理单个episode（单臂），包含视频帧（较慢）

    Args:
        episode: EpisodeStateFiles对象
        global_task_info: 全局task_info.json内容
        global_video_prompt: 全局video_prompt.txt内容

    Returns:
        (episode_data, tasks, video_prompt) 元组
    """
    try:
        # 1. 加载状态文件
        states_data = read_jsonl_file(episode.states_file) if episode.states_file else []

        # 2. 加载视频帧的辅助函数
        def load_video_frames(video_path: Optional[str]) -> List[np.ndarray]:
            frames = []
            if video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (256, 256))
                    frames.append(frame_resized)
                cap.release()
            return frames

        main_frames = load_video_frames(episode.main_camera_video)
        wrist_frames = load_video_frames(episode.wrist_camera_video)
        scene_frames = load_video_frames(episode.scene_camera_video)

        # 3. 从全局元数据获取任务信息
        tasks = ""
        if global_task_info:
            if isinstance(global_task_info, list) and len(global_task_info) > 0:
                if 'task_desc' in global_task_info[0] and 'prompt' in global_task_info[0]['task_desc']:
                    tasks = global_task_info[0]['task_desc']['prompt']
                elif 'task' in global_task_info[0]:
                    tasks = global_task_info[0]['task']
            elif isinstance(global_task_info, dict):
                if 'task_desc' in global_task_info and 'prompt' in global_task_info['task_desc']:
                    tasks = global_task_info['task_desc']['prompt']
                elif 'task' in global_task_info:
                    tasks = global_task_info['task']

        # 4. 从全局获取视频提示
        video_prompt = global_video_prompt if global_video_prompt else ""

        # 5. 生成episode数据
        states_len = len(states_data)

        if states_len < 2:
            print(f"  警告: Episode {episode.episode_num} 数据长度不足")
            return ([], tasks, video_prompt)

        valid_len = states_len - 1
        episode_data = []

        for index in range(valid_len):
            eef_state = np.array(states_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
            eef_action = np.array(states_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
            joints_state = np.array(states_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
            joints_action = np.array(states_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
            gripper_state = float(states_data[index].get('gripper', 0.0))
            gripper_action = float(states_data[index + 1].get('gripper', 0.0))

            # state: eef(7) + joints(6) + gripper(1) = 14
            state = np.concatenate((
                eef_state,
                joints_state,
                np.array([gripper_state], dtype=np.float32),
            ))

            # action: eef(7) + joints(6) + gripper(1) = 14
            action = np.concatenate((
                eef_action,
                joints_action,
                np.array([gripper_action], dtype=np.float32),
            ))

            frame_item = {
                'top': main_frames[index] if main_frames and index < len(main_frames) else None,
                'wrist': wrist_frames[index] if wrist_frames and index < len(wrist_frames) else None,
                'scene': scene_frames[index] if scene_frames and index < len(scene_frames) else None,
            }

            episode_data.append({
                'action': action,
                'state': state,
                'frames': frame_item,
            })

        return (episode_data, tasks, video_prompt)

    except Exception as e:
        print(f"错误处理 episode {episode.episode_num}: {e}")
        import traceback
        traceback.print_exc()
        return ([], "", "")


def process_episode_batch(episodes: List[EpisodeStateFiles], include_frames: bool = False,
                          global_task_info: Optional[Dict] = None,
                          global_video_prompt: Optional[str] = None) -> List[Tuple[List[Dict[str, Any]], str, str]]:
    """
    批量处理episode

    Args:
        episodes: EpisodeStateFiles对象列表
        include_frames: 是否包含视频帧
        global_task_info: 全局task_info.json内容
        global_video_prompt: 全局video_prompt.txt内容

    Returns:
        处理结果列表
    """
    if include_frames:
        process_func = partial(process_episode_with_frames,
                               global_task_info=global_task_info,
                               global_video_prompt=global_video_prompt)
    else:
        process_func = partial(process_episode_fast,
                               global_task_info=global_task_info,
                               global_video_prompt=global_video_prompt)

    return [process_func(ep) for ep in episodes]


# Module-level wrappers required for pool.imap (closures/lambdas can't be pickled)
def _process_fast_wrapper(args):
    episode, task_info, video_prompt = args
    return process_episode_fast(episode, task_info, video_prompt)


def _process_frames_wrapper(args):
    episode, task_info, video_prompt = args
    return process_episode_with_frames(episode, task_info, video_prompt)


def main(repo_name: str,
         data_dir: str = DATA_DIR, 
         robot_type: Optional[str] = None,
         include_frames: bool = True, num_episodes: Optional[int] = None,
         cameras_csv: str = DEFAULT_CAMERAS_CSV):
    """
    主函数

    Args:
        data_dir: 数据目录路径
        repo_name: Hugging Face仓库名称
        robot_type: 机器人类型（如 ARX5, UR5, Franka），用于从cameras.csv查找摄像头配置。
                    若不指定，则自动从第一个数据集的meta/task_info.json中检测。
        include_frames: 是否包含视频帧（如果为False，只处理状态和动作，速度更快）
        num_episodes: 处理的episode数量，None表示全部
        cameras_csv: cameras.csv文件路径
    """
    # 若未指定robot_type，则从第一个数据集的task_info.json中自动检测
    if robot_type is None:
        known_single_arm_types = get_single_arm_robot_types(cameras_csv)
        print(f"已知单臂机器人类型: {known_single_arm_types}")

        first_dataset_path = os.path.join(data_dir, RAW_DATASET_NAMES[0])
        _, _, first_task_info, _ = load_global_metadata(first_dataset_path)

        if first_task_info is None:
            raise ValueError(
                f"无法从 {first_dataset_path}/meta/task_info.json 加载元数据，"
                "请手动通过 --robot-type 参数指定机器人类型。"
            )

        robot_type = detect_robot_type_from_task_info(first_task_info, known_single_arm_types)

        if robot_type is None:
            raise ValueError(
                f"无法从 task_info.json 的 task_tag 中识别机器人类型。"
                f"已知类型: {known_single_arm_types}。"
                "请手动通过 --robot-type 参数指定。"
            )

        print(f"自动检测到机器人类型: {robot_type}")

    # 加载摄像头配置
    camera_config = load_camera_config(cameras_csv, robot_type)
    print(f"摄像头配置 ({robot_type}): main={camera_config['main']}, "
          f"wrist={camera_config['wrist']}, scene={camera_config['scene']}")

    # 清理输出目录
    os.system("export HF_LEROBOT_HOME='/liujinxin/code/lhc/lingbot-va/datasets/robochallenge'")
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # 设置缓存
    cache_path = os.path.join(data_dir, "cache")
    os.makedirs(cache_path, exist_ok=True)
    os.environ['HF_DATASETS_CACHE'] = cache_path

    print("=" * 60)
    print(f"开始处理数据集: {repo_name}")
    print(f"数据目录: {data_dir}")
    print(f"机器人类型: {robot_type}")
    print(f"包含视频帧: {include_frames}")
    print("=" * 60)

    start_time = time.time()

    # 构建视频特征字典
    video_feature = {
        "dtype": "video",
        "shape": (3, 256, 256),
        "names": ["rgb", "height", "width"],
        "info": {
            "video.height": 256,
            "video.width": 256,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": False,
        }
    }

    features = {
        "observation.images.top": video_feature,
        "observation.images.wrist": video_feature,
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["motors"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["motors"],
        },
    }

    # 如果有场景摄像头，添加到特征中
    if camera_config['scene']:
        features["observation.images.scene"] = video_feature

    # 创建LeRobot数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=30,
        features=features,
        use_videos=True,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 存储所有episode
    all_episodes = []

    known_single_arm_types = get_single_arm_robot_types(cameras_csv)

    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"\n处理数据集: {raw_dataset_name}")
        raw_dataset_path = os.path.join(data_dir, raw_dataset_name)

        # 加载全局元数据
        global_task_info_path, global_video_prompt_path, global_task_info, global_video_prompt = load_global_metadata(raw_dataset_path)

        # 检查当前数据集的机器人类型是否与预期一致
        if global_task_info is not None:
            detected = detect_robot_type_from_task_info(global_task_info, known_single_arm_types)
            if detected is not None and detected.lower() != robot_type.lower():
                print(f"  ⚠ 警告: 数据集 '{raw_dataset_name}' 中检测到机器人类型 '{detected}'，"
                      f"与当前使用的 '{robot_type}' 不一致，摄像头配置可能不匹配！")

        # 查找episode文件
        dataset_episodes = find_all_episode_state_files(raw_dataset_path, camera_config)

        all_episodes.extend(dataset_episodes)

        if not hasattr(main, 'global_metadata'):
            main.global_metadata = {}
        main.global_metadata[raw_dataset_name] = {
            'task_info': global_task_info,
            'video_prompt': global_video_prompt,
        }

    if not all_episodes:
        print('\n未找到任何状态文件，请检查目录结构是否正确。')
        return

    # 统计信息
    complete_episodes = [ep for ep in all_episodes if ep.is_complete()]
    total_files = sum(len(ep.get_file_paths()) for ep in all_episodes)

    print(f"\n详细统计:")
    print(f"  总episode数: {len(all_episodes)}")
    print(f"  完整episode数: {len(complete_episodes)}")
    print(f"  总文件数: {total_files}")

    # shuffled_episodes = shuffle_episode_state_files(all_episodes, seed=42)
    shuffled_episodes = all_episodes

    if num_episodes is not None and num_episodes > 0:
        shuffled_episodes = shuffled_episodes[:num_episodes]
        print(f"\n限制处理 {num_episodes} 个episode")

    print_episode_info(shuffled_episodes, "\n待处理的Episode列表")

    if include_frames:
        process_func = process_episode_with_frames
        print("\n使用完整模式（包含视频帧）...")
    else:
        process_func = process_episode_fast
        print("\n使用快速模式（仅状态和动作）...")

    num_processes = min(cpu_count(), NUM_WORKERS)
    print(f"使用 {num_processes} 个进程并行处理...")

    process_args = []
    for episode in shuffled_episodes:
        dataset_name = episode.dataset_name
        metadata = main.global_metadata.get(dataset_name, {'task_info': None, 'video_prompt': None})
        process_args.append((episode, metadata['task_info'], metadata['video_prompt']))

    wrapper = _process_frames_wrapper if include_frames else _process_fast_wrapper

    # Use imap (chunksize=1) instead of starmap so each episode's result is
    # consumed and freed immediately after writing, rather than accumulating
    # all NUM_WORKERS results in memory at once.
    with Pool(processes=num_processes) as pool:
        for i, (episode_data, tasks, video_prompt) in enumerate(
            pool.imap(wrapper, process_args, chunksize=1)
        ):
            episode = shuffled_episodes[i]

            if not episode_data:
                print(f"  [{i+1}/{len(process_args)}] 跳过空的episode {episode.episode_num}")
                continue

            print(f"  [{i+1}/{len(process_args)}] 添加episode {episode.episode_num}: {len(episode_data)}个时间步")

            for frame_data in episode_data:
                frame_dict = {
                    'action': frame_data['action'],
                    'observation.state': frame_data['state'],
                }

                if include_frames and 'frames' in frame_data:
                    frame_dict['observation.images.top'] = frame_data['frames'].get('top')
                    frame_dict['observation.images.wrist'] = frame_data['frames'].get('wrist')
                    if camera_config['scene']:
                        frame_dict['observation.images.scene'] = frame_data['frames'].get('scene')

                dataset.add_frame(frame_dict, task=tasks)

            dataset.save_episode()

    total_time = time.time() - start_time
    print(f"\n所有episode处理完成，总耗时: {total_time:.2f}秒")

    # dataset.consolidate(run_compute_stats=True)
    # if PUSH_TO_HUB:
    #     print(f"\n推送到Hugging Face Hub: {repo_name}")
    #     dataset.push_to_hub(
    #         tags=["single_arm", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )
    #     print("推送完成")

    print(f"\n数据集已保存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
