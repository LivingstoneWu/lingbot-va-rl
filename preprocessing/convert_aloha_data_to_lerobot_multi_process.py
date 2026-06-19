import os
import json
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import glob
import threading
import queue
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import tyro
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置常量
HF_LEROBOT_HOME = Path("/luhongchao/shared/dataset/robochallenge_v2")  # 请修改为实际路径
RAW_DATASET_NAMES = ["press_three_buttons"]  # 请修改为实际数据集名称, original dataset name
PUSH_TO_HUB = False
PROCESS_BATCH_SIZE=8
NUM_WORKERS=4
PREFETCH_QUEUE_SIZE=32
IMAGE_WRITER_PROCESSES=0
DATA_DIR = "/luhongchao/shared/dataset/robochallenge_v2_raw"

class EpisodeStateFiles:
    """表示一个episode的左右状态文件和相关资源"""
    def __init__(self, episode_num: str, episode_dir: str, dataset_name: str = "", 
                 global_task_info: Optional[str] = None, global_video_prompt: Optional[str] = None):
        self.episode_num = episode_num
        self.episode_dir = episode_dir
        self.dataset_name = dataset_name
        self.left_file: Optional[str] = None
        self.right_file: Optional[str] = None
        self.states_dir: Optional[str] = None
        self.videos_dir: Optional[str] = None
        
        # 特定视频文件
        self.cam_high_video: Optional[str] = None      # cam_high_rgb.mp4
        self.cam_wrist_left_video: Optional[str] = None # cam_wrist_left_rgb.mp4
        self.cam_wrist_right_video: Optional[str] = None # cam_wrist_right_rgb.mp4
        
        # 全局元数据文件（所有episode共享）
        self.global_task_info_file = global_task_info
        self.global_video_prompt_file = global_video_prompt
    
    def is_complete(self) -> bool:
        """检查是否同时包含左右文件"""
        return self.left_file is not None and self.right_file is not None
    
    def has_all_videos(self) -> bool:
        """检查是否包含所有三个视频文件"""
        return (self.cam_high_video is not None and 
                self.cam_wrist_left_video is not None and 
                self.cam_wrist_right_video is not None)
    
    def has_task_info(self) -> bool:
        """检查是否有全局task_info.json"""
        return self.global_task_info_file is not None and os.path.exists(self.global_task_info_file)
    
    def has_video_prompt(self) -> bool:
        """检查是否有全局video_prompt.txt"""
        return self.global_video_prompt_file is not None and os.path.exists(self.global_video_prompt_file)
    
    def get_video_dict(self) -> Dict[str, Optional[str]]:
        """获取所有视频文件的字典"""
        return {
            'cam_high': self.cam_high_video,
            'cam_wrist_left': self.cam_wrist_left_video,
            'cam_wrist_right': self.cam_wrist_right_video
        }
    
    def get_files_dict(self) -> Dict[str, Any]:
        """获取所有文件的字典"""
        return {
            'left': self.left_file,
            'right': self.right_file,
            'videos': self.get_video_dict(),
            'global_task_info': self.global_task_info_file,
            'global_video_prompt': self.global_video_prompt_file
        }
    
    def get_file_paths(self) -> List[str]:
        """获取所有存在的文件路径列表"""
        paths = []
        if self.left_file:
            paths.append(self.left_file)
        if self.right_file:
            paths.append(self.right_file)
        if self.cam_high_video:
            paths.append(self.cam_high_video)
        if self.cam_wrist_left_video:
            paths.append(self.cam_wrist_left_video)
        if self.cam_wrist_right_video:
            paths.append(self.cam_wrist_right_video)
        if self.global_task_info_file:
            paths.append(self.global_task_info_file)
        if self.global_video_prompt_file:
            paths.append(self.global_video_prompt_file)
        return paths
    
    def __repr__(self) -> str:
        status = "完整" if self.is_complete() else "不完整"
        video_status = "完整视频" if self.has_all_videos() else f"部分视频({sum(1 for v in [self.cam_high_video, self.cam_wrist_left_video, self.cam_wrist_right_video] if v)}/3)"
        return f"EpisodeStateFiles(episode={self.episode_num}, status={status}, {video_status})"

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
                if line:  # 跳过空行
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
    
    # 检查并读取task_info.json
    if os.path.exists(task_info_path):
        task_info_content = read_json_file(task_info_path)
        if task_info_content:
            print(f"✓ 加载全局 task_info.json: {task_info_path}")
    else:
        print(f"⚠ 全局 task_info.json 不存在: {task_info_path}")
        task_info_path = None
    
    # 检查并读取video_prompt.txt
    if os.path.exists(video_prompt_path):
        video_prompt_content = read_text_file(video_prompt_path)
        if video_prompt_content:
            print(f"✓ 加载全局 video_prompt.txt: {video_prompt_path}")
    else:
        print(f"⚠ 全局 video_prompt.txt 不存在: {video_prompt_path}")
        video_prompt_path = None
    
    return task_info_path, video_prompt_path, task_info_content, video_prompt_content

def find_all_episode_state_files(base_dir: str) -> List[EpisodeStateFiles]:
    """
    查找所有episode的左右状态文件和相关资源，以episode为单位组织
    
    Args:
        base_dir: 基础目录名
    
    Returns:
        EpisodeStateFiles对象列表，每个对象包含一个episode的所有文件信息
    """
    episode_files_map = {}
    dataset_name = os.path.basename(base_dir)
    
    # 确保基础目录存在
    if not os.path.exists(base_dir):
        print(f"警告：目录 '{base_dir}' 不存在")
        return []
    
    # 加载全局元数据
    global_task_info_path, global_video_prompt_path, _, _ = load_global_metadata(base_dir)
    
    # 遍历所有episode目录
    episode_pattern = os.path.join(base_dir, "data", "episode_*")
    episode_dirs = glob.glob(episode_pattern)
    
    print(f"在 {dataset_name} 中找到 {len(episode_dirs)} 个episode目录")
    
    for episode_dir in sorted(episode_dirs):
        episode_num = os.path.basename(episode_dir).replace("episode_", "")
        
        # 创建EpisodeStateFiles对象，传入全局元数据路径
        episode = EpisodeStateFiles(
            episode_num, 
            episode_dir, 
            dataset_name,
            global_task_info=global_task_info_path,
            global_video_prompt=global_video_prompt_path
        )
        
        # 设置各个子目录
        episode.states_dir = os.path.join(episode_dir, "states")
        episode.videos_dir = os.path.join(episode_dir, "videos")
        
        # 1. 检查states目录和文件
        if os.path.exists(episode.states_dir):
            left_path = os.path.join(episode.states_dir, "left_states.jsonl")
            right_path = os.path.join(episode.states_dir, "right_states.jsonl")
            
            if os.path.exists(left_path):
                episode.left_file = left_path
            
            if os.path.exists(right_path):
                episode.right_file = right_path
        
        # 2. 检查videos目录和特定视频文件
        if os.path.exists(episode.videos_dir):
            cam_high_path = os.path.join(episode.videos_dir, "cam_high_rgb.mp4")
            cam_wrist_left_path = os.path.join(episode.videos_dir, "cam_left_wrist_rgb.mp4")
            cam_wrist_right_path = os.path.join(episode.videos_dir, "cam_right_wrist_rgb.mp4")
            
            if os.path.exists(cam_high_path):
                episode.cam_high_video = cam_high_path
            
            if os.path.exists(cam_wrist_left_path):
                episode.cam_wrist_left_video = cam_wrist_left_path
            
            if os.path.exists(cam_wrist_right_path):
                episode.cam_wrist_right_video = cam_wrist_right_path
        
        episode_files_map[episode_num] = episode
    
    # 转换为列表
    episode_files_list = list(episode_files_map.values())
    
    return episode_files_list

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
        
        # 视频状态
        video_files = []
        if episode.cam_high_video:
            video_files.append("H")
        if episode.cam_wrist_left_video:
            video_files.append("WL")
        if episode.cam_wrist_right_video:
            video_files.append("WR")
        video_status = f"{len(video_files)}/3 ({','.join(video_files)})" if video_files else "无"
        
        dataset_short = episode.dataset_name[:15] if len(episode.dataset_name) > 15 else episode.dataset_name
        
        print(f"{i:<4} {dataset_short:<15} episode_{episode.episode_num:<6} {status:<8} {video_status:<30}")
    
    print("-" * 100)

def pad_joint_dimension(data_array: np.ndarray, expected_dim: int = 7) -> np.ndarray:
    """
    如果关节维度为6，则添加一位padding为0，变为7
    
    Args:
        data_array: 输入的数据数组
        expected_dim: 期望的维度，默认为7
    
    Returns:
        处理后的数组，维度为expected_dim
    """
    if data_array.shape[0] == 6 and expected_dim == 7:
        # 如果是6维，添加一个0 padding
        return np.concatenate([data_array, np.zeros(1, dtype=data_array.dtype)])
    elif data_array.shape[0] == expected_dim:
        return data_array
    else:
        print(f"警告：意外的数组维度 {data_array.shape[0]}，期望 {expected_dim}，将使用零填充")
        padded = np.zeros(expected_dim, dtype=data_array.dtype)
        min_dim = min(data_array.shape[0], expected_dim)
        padded[:min_dim] = data_array[:min_dim]
        return padded

def process_episode_fast(episode: EpisodeStateFiles, global_task_info: Optional[Dict] = None, 
                        global_video_prompt: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    快速处理单个episode，优化速度
    
    Args:
        episode: EpisodeStateFiles对象
        global_task_info: 全局task_info.json内容
        global_video_prompt: 全局video_prompt.txt内容
    
    Returns:
        (episode_data, tasks, video_prompt) 元组
    """
    try:
        # 1. 快速加载状态文件
        left_data = []
        right_data = []
        
        if episode.left_file and os.path.exists(episode.left_file):
            left_data = read_jsonl_file(episode.left_file)
        
        if episode.right_file and os.path.exists(episode.right_file):
            right_data = read_jsonl_file(episode.right_file)
        
        # 2. 获取视频信息（不加载帧）
        cam_high_frames_count, cam_high_shape = read_video_info(episode.cam_high_video)
        cam_wrist_left_frames_count, cam_wrist_left_shape = read_video_info(episode.cam_wrist_left_video)
        cam_wrist_right_frames_count, cam_wrist_right_shape = read_video_info(episode.cam_wrist_right_video)
        
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
        episode_data = []
        
        left_len = len(left_data)
        right_len = len(right_data)
        
        if left_len < 2 and right_len < 2:
            print(f"  警告: Episode {episode.episode_num} 数据长度不足")
            return ([], tasks, video_prompt)
        
        min_len = min(left_len, right_len) if left_len > 0 and right_len > 0 else max(left_len, right_len)
        valid_len = min_len - 1 if min_len > 1 else 0
        
        # 预分配列表以提高性能
        for index in range(valid_len):
            # 处理左侧数据
            if index < left_len - 1:
                # 获取原始数据
                left_eef_state_raw = np.array(left_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                left_eef_action_raw = np.array(left_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                left_joints_state_raw = np.array(left_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
                left_joints_action_raw = np.array(left_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
                left_gripper_state = float(left_data[index].get('gripper', 0.0))
                left_gripper_action = float(left_data[index + 1].get('gripper', 0.0))
                
                # 处理关节维度（如果是6维，padding为7）
                # left_eef_state = pad_joint_dimension(left_eef_state_raw)
                # left_eef_action = pad_joint_dimension(left_eef_action_raw)
                # left_joints_state = pad_joint_dimension(left_joints_state_raw)
                # left_joints_action = pad_joint_dimension(left_joints_action_raw)
                left_eef_state = left_eef_state_raw
                left_eef_action = left_eef_action_raw
                left_joints_state = left_joints_state_raw
                left_joints_action = left_joints_action_raw
            else:
                left_eef_state = np.zeros(7, dtype=np.float32)
                left_eef_action = np.zeros(7, dtype=np.float32)
                left_joints_state = np.zeros(6, dtype=np.float32)
                left_gripper_state = 0.0
                left_joints_action = np.zeros(6, dtype=np.float32)
                left_gripper_action = 0.0
            
            # 处理右侧数据
            if index < right_len - 1:
                # 获取原始数据
                right_eef_state_raw = np.array(right_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                right_eef_action_raw = np.array(right_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                right_joints_state_raw = np.array(right_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
                right_joints_action_raw = np.array(right_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
                right_gripper_state = float(right_data[index].get('gripper', 0.0))
                right_gripper_action = float(right_data[index + 1].get('gripper', 0.0))
                
                # 处理关节维度（如果是6维，padding为7）
                # right_eef_state = pad_joint_dimension(right_eef_state_raw)
                # right_eef_action = pad_joint_dimension(right_eef_action_raw)
                # right_joints_state = pad_joint_dimension(right_joints_state_raw)
                # right_joints_action = pad_joint_dimension(right_joints_action_raw)
                right_eef_state = right_eef_state_raw
                right_eef_action = right_eef_action_raw
                right_joints_state = right_joints_state_raw
                right_joints_action = right_joints_action_raw
            else:
                right_eef_state = np.zeros(7, dtype=np.float32)
                right_eef_action = np.zeros(7, dtype=np.float32)
                right_joints_state = np.zeros(6, dtype=np.float32)
                right_gripper_state = 0.0
                right_joints_action = np.zeros(6, dtype=np.float32)
                right_gripper_action = 0.0
            
            # 合并状态和动作
            state = np.concatenate((
                # left_eef_state, right_eef_state,
                left_joints_state, right_joints_state,
                np.array([left_gripper_state, right_gripper_state], dtype=np.float32)
            ))
            
            action = np.concatenate((
                # left_eef_action, right_eef_action,
                left_joints_action, right_joints_action,
                np.array([left_gripper_action, right_gripper_action], dtype=np.float32)
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
    处理单个episode，包含视频帧（较慢）
    
    Args:
        episode: EpisodeStateFiles对象
        global_task_info: 全局task_info.json内容
        global_video_prompt: 全局video_prompt.txt内容
    
    Returns:
        (episode_data, tasks, video_prompt) 元组
    """
    try:
        # 1. 加载状态文件
        left_data = read_jsonl_file(episode.left_file) if episode.left_file else []
        right_data = read_jsonl_file(episode.right_file) if episode.right_file else []
        
        # 2. 加载视频帧
        cam_high_frames = []
        cam_wrist_left_frames = []
        cam_wrist_right_frames = []
        
        if episode.cam_high_video and os.path.exists(episode.cam_high_video):
            cap = cv2.VideoCapture(episode.cam_high_video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (256, 256))
                cam_high_frames.append(frame_resized)
            cap.release()
        
        if episode.cam_wrist_left_video and os.path.exists(episode.cam_wrist_left_video):
            cap = cv2.VideoCapture(episode.cam_wrist_left_video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (256, 256))
                cam_wrist_left_frames.append(frame_resized)
            cap.release()
        
        if episode.cam_wrist_right_video and os.path.exists(episode.cam_wrist_right_video):
            cap = cv2.VideoCapture(episode.cam_wrist_right_video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (256, 256))
                cam_wrist_right_frames.append(frame_resized)
            cap.release()
        
        # 3. 从全局元数据获取任务信息
        tasks = ""
        if global_task_info:
            if isinstance(global_task_info, list) and len(global_task_info) > 0:
                if 'task_desc' in global_task_info[0] and 'prompt' in global_task_info[0]['task_desc']:
                    tasks = global_task_info[0]['task_desc']['prompt']
            elif isinstance(global_task_info, dict):
                if 'task_desc' in global_task_info and 'prompt' in global_task_info['task_desc']:
                    tasks = global_task_info['task_desc']['prompt']
        
        # 4. 从全局获取视频提示
        video_prompt = global_video_prompt if global_video_prompt else ""
        
        # 5. 生成episode数据
        episode_data = []
        
        left_len = len(left_data)
        right_len = len(right_data)
        min_len = min(left_len, right_len) if left_len > 0 and right_len > 0 else max(left_len, right_len)
        valid_len = min_len - 1 if min_len > 1 else 0
        
        for index in range(valid_len):
            # 处理左侧数据
            if index < left_len - 1:
                # 获取原始数据
                left_eef_state_raw = np.array(left_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                left_eef_action_raw = np.array(left_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                left_joints_state_raw = np.array(left_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
                left_joints_action_raw = np.array(left_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
                left_gripper_state = float(left_data[index].get('gripper', 0.0))
                left_gripper_action = float(left_data[index + 1].get('gripper', 0.0))
                
                # 处理关节维度（如果是6维，padding为7）
                # left_eef_state = pad_joint_dimension(left_eef_state_raw)
                # left_eef_action = pad_joint_dimension(left_eef_action_raw)
                # left_joints_state = pad_joint_dimension(left_joints_state_raw)
                # left_joints_action = pad_joint_dimension(left_joints_action_raw)
                left_eef_state = left_eef_state_raw
                left_eef_action = left_eef_action_raw
                left_joints_state = left_joints_state_raw
                left_joints_action = left_joints_action_raw
            else:
                left_eef_state = np.zeros(7, dtype=np.float32)
                left_eef_action = np.zeros(7, dtype=np.float32)
                left_joints_state = np.zeros(6, dtype=np.float32)
                left_gripper_state = 0.0
                left_joints_action = np.zeros(6, dtype=np.float32)
                left_gripper_action = 0.0
            
            # 处理右侧数据
            if index < right_len - 1:
                # 获取原始数据
                right_eef_state_raw = np.array(right_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                right_eef_action_raw = np.array(right_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                right_joints_state_raw = np.array(right_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
                right_joints_action_raw = np.array(right_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
                right_gripper_state = float(right_data[index].get('gripper', 0.0))
                right_gripper_action = float(right_data[index + 1].get('gripper', 0.0))
                
                # 处理关节维度（如果是6维，padding为7）
                # right_eef_state = pad_joint_dimension(right_eef_state_raw)
                # right_eef_action = pad_joint_dimension(right_eef_action_raw)
                # right_joints_state = pad_joint_dimension(right_joints_state_raw)
                # right_joints_action = pad_joint_dimension(right_joints_action_raw)
                right_eef_state = right_eef_state_raw
                right_eef_action = right_eef_action_raw
                right_joints_state = right_joints_state_raw
                right_joints_action = right_joints_action_raw
            else:
                right_eef_state = np.zeros(7, dtype=np.float32)
                right_eef_action = np.zeros(7, dtype=np.float32)
                right_joints_state = np.zeros(6, dtype=np.float32)
                right_gripper_state = 0.0
                right_joints_action = np.zeros(6, dtype=np.float32)
                right_gripper_action = 0.0
            
            # 合并状态和动作
            state = np.concatenate((
                # left_eef_state, right_eef_state,
                left_joints_state, right_joints_state,
                np.array([left_gripper_state, right_gripper_state], dtype=np.float32)
            ))
            
            action = np.concatenate((
                # left_eef_action, right_eef_action,
                left_joints_action, right_joints_action,
                np.array([left_gripper_action, right_gripper_action], dtype=np.float32)
            ))
            
            # 添加视频帧
            frame_item = {}
            if cam_high_frames and index < len(cam_high_frames):
                frame_item['cam_high'] = cam_high_frames[index]
            else:
                frame_item['cam_high'] = None
            
            if cam_wrist_left_frames and index < len(cam_wrist_left_frames):
                frame_item['cam_wrist_left'] = cam_wrist_left_frames[index]
            else:
                frame_item['cam_wrist_left'] = None
            
            if cam_wrist_right_frames and index < len(cam_wrist_right_frames):
                frame_item['cam_wrist_right'] = cam_wrist_right_frames[index]
            else:
                frame_item['cam_wrist_right'] = None
            
            episode_data.append({
                'action': action,
                'state': state,
                'frames': frame_item
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
    results = []
    
    if include_frames:
        process_func = partial(process_episode_with_frames, 
                              global_task_info=global_task_info, 
                              global_video_prompt=global_video_prompt)
    else:
        process_func = partial(process_episode_fast, 
                              global_task_info=global_task_info, 
                              global_video_prompt=global_video_prompt)
    
    for episode in episodes:
        result = process_func(episode)
        results.append(result)
    
    return results


def process_episode_stream_to_dataset(
    episode: EpisodeStateFiles,
    dataset: LeRobotDataset,
    global_task_info: Optional[Dict] = None,
    global_video_prompt: Optional[str] = None,
) -> int:
    """
    逐帧流式处理单个episode并直接写入LeRobotDataset。
    避免一次性加载整段视频和跨进程传输大量帧数据。

    Returns:
        写入的数据步数
    """
    try:
        left_data = read_jsonl_file(episode.left_file) if episode.left_file else []
        right_data = read_jsonl_file(episode.right_file) if episode.right_file else []

        left_len = len(left_data)
        right_len = len(right_data)
        min_len = min(left_len, right_len) if left_len > 0 and right_len > 0 else max(left_len, right_len)
        valid_len = min_len - 1 if min_len > 1 else 0
        if valid_len <= 0:
            return 0

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

        _ = global_video_prompt  # 目前仅保留接口，不影响写入逻辑

        caps = {
            "cam_high": cv2.VideoCapture(episode.cam_high_video) if episode.cam_high_video and os.path.exists(episode.cam_high_video) else None,
            "cam_wrist_left": cv2.VideoCapture(episode.cam_wrist_left_video) if episode.cam_wrist_left_video and os.path.exists(episode.cam_wrist_left_video) else None,
            "cam_wrist_right": cv2.VideoCapture(episode.cam_wrist_right_video) if episode.cam_wrist_right_video and os.path.exists(episode.cam_wrist_right_video) else None,
        }

        frame_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
        sentinel = {"__done__": True}
        stop_event = threading.Event()

        def _producer() -> None:
            try:
                cam_to_feature = {
                    "cam_high": "observation.images.top",
                    "cam_wrist_left": "observation.images.left_wrist",
                    "cam_wrist_right": "observation.images.right_wrist",
                }
                for _ in range(valid_len):
                    if stop_event.is_set():
                        break
                    item: Dict[str, Any] = {}
                    for cam_key, feature_key in cam_to_feature.items():
                        if stop_event.is_set():
                            break
                        cap = caps[cam_key]
                        if cap is None:
                            item[feature_key] = None
                            continue
                        ret, frame = cap.read()
                        if not ret:
                            item[feature_key] = None
                            continue
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        item[feature_key] = cv2.resize(frame_rgb, (256, 256))
                    while not stop_event.is_set():
                        try:
                            frame_queue.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                while not stop_event.is_set():
                    try:
                        frame_queue.put({"__error__": str(exc)}, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            finally:
                while not stop_event.is_set():
                    try:
                        frame_queue.put(sentinel, timeout=0.2)
                        break
                    except queue.Full:
                        continue

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        try:
            for index in range(valid_len):
                if index < left_len - 1:
                    left_eef_state = np.array(left_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                    left_eef_action = np.array(left_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                    left_joints_state = np.array(left_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
                    left_joints_action = np.array(left_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
                    left_gripper_state = float(left_data[index].get('gripper', 0.0))
                    left_gripper_action = float(left_data[index + 1].get('gripper', 0.0))
                else:
                    left_eef_state = np.zeros(7, dtype=np.float32)
                    left_eef_action = np.zeros(7, dtype=np.float32)
                    left_joints_state = np.zeros(6, dtype=np.float32)
                    left_gripper_state = 0.0
                    left_joints_action = np.zeros(6, dtype=np.float32)
                    left_gripper_action = 0.0

                if index < right_len - 1:
                    right_eef_state = np.array(right_data[index].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                    right_eef_action = np.array(right_data[index + 1].get('ee_pose_quaternion', np.zeros(7)), dtype=np.float32)
                    right_joints_state = np.array(right_data[index].get('joint_positions', np.zeros(6)), dtype=np.float32)
                    right_joints_action = np.array(right_data[index + 1].get('joint_positions', np.zeros(6)), dtype=np.float32)
                    right_gripper_state = float(right_data[index].get('gripper', 0.0))
                    right_gripper_action = float(right_data[index + 1].get('gripper', 0.0))
                else:
                    right_eef_state = np.zeros(7, dtype=np.float32)
                    right_eef_action = np.zeros(7, dtype=np.float32)
                    right_joints_state = np.zeros(6, dtype=np.float32)
                    right_gripper_state = 0.0
                    right_joints_action = np.zeros(6, dtype=np.float32)
                    right_gripper_action = 0.0

                state = np.concatenate((
                    # left_eef_state, right_eef_state,
                    left_joints_state, right_joints_state,
                    np.array([left_gripper_state, right_gripper_state], dtype=np.float32)
                ))
                action = np.concatenate((
                    # left_eef_action, right_eef_action,
                    left_joints_action, right_joints_action,
                    np.array([left_gripper_action, right_gripper_action], dtype=np.float32)
                ))

                frame_dict = {
                    "action": action,
                    "observation.state": state,
                }

                frame_item = frame_queue.get()
                if frame_item is sentinel or frame_item.get("__done__"):
                    break
                if "__error__" in frame_item:
                    raise RuntimeError(f"Frame producer failed: {frame_item['__error__']}")
                frame_dict.update(frame_item)

                dataset.add_frame(frame_dict, task=tasks)

            dataset.save_episode()
        finally:
            stop_event.set()
            producer.join(timeout=2.0)
            for cap in caps.values():
                if cap is not None:
                    cap.release()

        return valid_len
    except Exception as e:
        print(f"错误处理 episode {episode.episode_num} (streaming): {e}")
        import traceback
        traceback.print_exc()
        return 0

def main(
    repo_name: str,
    raw_dataset: List[str] = RAW_DATASET_NAMES,
    data_dir: str = DATA_DIR,
    include_frames: bool = True,
    num_episodes: Optional[int] = None,
):
    """
    主函数
    
    Args:
        data_dir: 数据目录路径
        repo_name: Hugging Face仓库名称
        include_frames: 是否包含视频帧（如果为False，只处理状态和动作，速度更快）
        num_episodes: 处理的episode数量，None表示全部
    """
    # 显式设置输出根目录（os.system("export ...") 不会影响当前Python进程）
    os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
    lerobot_dataset_module.HF_LEROBOT_HOME = HF_LEROBOT_HOME

    # 清理输出目录
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
    print(f"原始数据集: {raw_dataset}")
    print(f"包含视频帧: {include_frames}")
    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 创建LeRobot数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="aloha",
        fps=30,
        features={
            "observation.images.top": {
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
            },
            "observation.images.left_wrist": {
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
            },
            "observation.images.right_wrist": {
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
            },
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
        },
        use_videos=True,
        image_writer_threads=10,
        image_writer_processes=IMAGE_WRITER_PROCESSES,
    )
    
    # 存储所有episode
    all_episodes = []
    
    if not raw_dataset:
        print("ERROR: raw_dataset 为空，请通过 --raw-dataset 指定至少一个数据集名称。")
        return

    # 对于每个数据集，先加载全局元数据，然后查找episode
    for raw_dataset_name in raw_dataset:
        print(f"\n处理数据集: {raw_dataset_name}")
        raw_dataset_path = os.path.join(data_dir, raw_dataset_name)
        
        # 加载全局元数据
        global_task_info_path, global_video_prompt_path, global_task_info, global_video_prompt = load_global_metadata(raw_dataset_path)
        
        # 查找episode文件
        dataset_episodes = find_all_episode_state_files(raw_dataset_path)
        
        # 为每个episode设置全局元数据（已经在构造函数中设置）
        all_episodes.extend(dataset_episodes)
        
        # 保存全局元数据供后续使用
        if not hasattr(main, 'global_metadata'):
            main.global_metadata = {}
        main.global_metadata[raw_dataset_name] = {
            'task_info': global_task_info,
            'video_prompt': global_video_prompt
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
    
    # 对episode列表进行shuffle
    # shuffled_episodes = shuffle_episode_state_files(all_episodes, seed=42)
    shuffled_episodes = all_episodes
    
    # 限制处理的episode数量
    if num_episodes is not None and num_episodes > 0:
        shuffled_episodes = shuffled_episodes[:num_episodes]
        print(f"\n限制处理 {num_episodes} 个episode")
    
    # 打印shuffle后的episode列表
    print_episode_info(shuffled_episodes, "\n待处理的Episode列表")
    
    if include_frames:
        print("\n使用流式完整模式（包含视频帧，低内存）...")
        total_steps = 0
        for episode_idx, episode in enumerate(shuffled_episodes, start=1):
            dataset_name = episode.dataset_name
            metadata = main.global_metadata.get(dataset_name, {'task_info': None, 'video_prompt': None})
            step_num = process_episode_stream_to_dataset(
                episode=episode,
                dataset=dataset,
                global_task_info=metadata['task_info'],
                global_video_prompt=metadata['video_prompt'],
            )
            if step_num <= 0:
                print(f"  跳过空的episode {episode.episode_num}")
                continue
            total_steps += step_num
            if episode_idx % 10 == 0 or episode_idx == len(shuffled_episodes):
                print(f"  已处理 {episode_idx}/{len(shuffled_episodes)} 个episode，累计时间步 {total_steps}")
    else:
        print("\n使用快速模式（仅状态和动作）...")
        # 使用多进程加速处理（仅在不含视频帧时）
        num_processes = min(cpu_count(), NUM_WORKERS)
        print(f"使用 {num_processes} 个进程并行处理...")

        process_args = []
        for episode in shuffled_episodes:
            dataset_name = episode.dataset_name
            metadata = main.global_metadata.get(dataset_name, {'task_info': None, 'video_prompt': None})
            process_args.append((episode, metadata['task_info'], metadata['video_prompt']))

        with Pool(processes=num_processes) as pool:
            batch_size = PROCESS_BATCH_SIZE
            for i in range(0, len(process_args), batch_size):
                batch_args = process_args[i:i+batch_size]
                batch_start_time = time.time()

                print(f"\n处理批次 {i//batch_size + 1}/{(len(process_args)-1)//batch_size + 1} ({len(batch_args)}个episode)")
                results = pool.starmap(process_episode_fast, batch_args)

                for episode_idx, (episode_data, tasks, video_prompt) in enumerate(results):
                    episode = shuffled_episodes[i + episode_idx]
                    if not episode_data:
                        print(f"  跳过空的episode {episode.episode_num}")
                        continue

                    print(f"  添加episode {episode.episode_num}: {len(episode_data)}个时间步")
                    for frame_data in episode_data:
                        frame_dict = {
                            'action': frame_data['action'],
                            'observation.state': frame_data['state'],
                        }
                        dataset.add_frame(frame_dict, task=tasks)
                    dataset.save_episode()

                batch_time = time.time() - batch_start_time
                print(f"  批次处理完成，耗时: {batch_time:.2f}秒")
    
    total_time = time.time() - start_time
    print(f"\n所有episode处理完成，总耗时: {total_time:.2f}秒")
    
    # dataset.consolidate(run_compute_stats=True)
    # 推送到Hugging Face Hub
    # if PUSH_TO_HUB:
    #     print(f"\n推送到Hugging Face Hub: {repo_name}")
    #     dataset.push_to_hub(
    #         tags=["libero", "bimanual", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )
    #     print("推送完成")
    
    print(f"\n数据集已保存到: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    tyro.cli(main)
