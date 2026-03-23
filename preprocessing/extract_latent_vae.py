import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS
from wan_va.modules.utils import load_vae, load_tokenizer, load_text_encoder
from wan_va.utils import init_logger, logger


class WanVAEEncoder:
    """Wan2.2 VAE Encoder - 用于批量编码视频帧"""
    
    def __init__(self, vae_path, device):
        self.device = device
        self.vae = load_vae(vae_path, torch_dtype=torch.float32, torch_device=device)
        self.vae_stride = (4, 8, 8)  # VAE的时空压缩倍数
        
        # 加载VAE的均值和标准差
        self.latents_mean = torch.tensor(self.vae.config.latents_mean)
        self.latents_std = torch.tensor(self.vae.config.latents_std)

        # 初始化video_processor用于后处理
        from diffusers.video_processor import VideoProcessor
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        
    def normalize_latents(self, latents, latents_mean, latents_std):
        """归一化latents"""
        # 确保latents是tensor
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(latents)}")
            
        # 获取latents的device
        device = latents.device
        
        # 调整均值和标准差的形状并移动到正确的device
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=device)
        
        # 归一化
        latents = ((latents.float() - latents_mean) * latents_std).to(latents.dtype)
        return latents
        
    def encode_video(self, video_tensor):
        """
        编码整个视频tensor
        
        Args:
            video_tensor: [3, F, H, W] 归一化到[-1, 1]
            
        Returns:
            latent: [C_lat, F_lat, H_lat, W_lat] 归一化后的latent
        """
        with torch.no_grad():
            # 添加batch维度 [1, C, F, H, W]
            video = video_tensor.unsqueeze(0).to(self.device)
            
            # VAE编码
            if hasattr(self.vae, 'encode'):
                enc_out = self.vae.encode(video)
                
                # 处理不同类型的输出
                if hasattr(enc_out, 'latent_dist'):
                    # AutoencoderKLOutput 类型
                    mu = enc_out.latent_dist.mean
                elif isinstance(enc_out, tuple):
                    # 如果是tuple，可能是 (mu, logvar)
                    mu, logvar = enc_out
                else:
                    # 其他情况
                    mu = enc_out
            else:
                # 使用streaming_vae
                from wan_va.modules.utils import WanVAEStreamingWrapper
                streaming_vae = WanVAEStreamingWrapper(self.vae)
                enc_out = streaming_vae.encode_chunk(video)
                mu, logvar = torch.chunk(enc_out, 2, dim=1)
            
            # 确保mu是tensor
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"Expected mu to be torch.Tensor, got {type(mu)}")
            
            # 对latent进行归一化
            latents_mean = self.latents_mean.to(mu.device)
            latents_std = self.latents_std.to(mu.device)
            mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
            
            # 去除batch维度
            latent = mu_norm.squeeze(0)
            # Flatten latent: [C, F_lat, H_lat, W_lat] -> [F_lat * H_lat * W_lat, C]
            C = latent.shape[0]
            F_lat = latent.shape[1]
            H_lat = latent.shape[2]
            W_lat = latent.shape[3]
            
            # 重新排列维度: [C, F, H, W] -> [F, H, W, C] -> [F*H*W, C]
            latent = latent.permute(1, 2, 3, 0).contiguous()  # [F_lat, H_lat, W_lat, C]
            latent = latent.view(-1, C)  # [F_lat * H_lat * W_lat, C]
            
            # 转移到CPU
            latent = latent.cpu()
            
        return latent, C, F_lat, H_lat, W_lat
    

class VideoProcessor:
    """视频处理器 - 负责帧采样和预处理"""
    
    def __init__(self, target_fps=10, target_height=256, target_width=256, chunk_size=450):
        self.target_fps = target_fps
        self.target_height = target_height
        self.target_width = target_width
        self.chunk_size = chunk_size  # 每个pth文件包含的帧数
        
        # 验证尺寸要求
        assert target_height % 32 == 0, f"Height must be multiple of 32, got {target_height}"
        assert target_width % 32 == 0, f"Width must be multiple of 32, got {target_width}"
    
    def process_video(self, video_path):
        """
        处理视频文件：采样帧、调整大小、返回所有帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            frames: list of PIL Images (所有采样帧)
            frame_ids: list of frame indices
            metadata: dict with fps, num_frames, etc.
        """
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样步长
        if self.target_fps >= orig_fps:
            step = 1
            actual_fps = orig_fps
        else:
            step = int(orig_fps / self.target_fps)
            actual_fps = orig_fps / step
        
        # 确定采样帧数
        target_frame_count = int(total_frames / step)
        
        # 调整帧数以满足 (num_frames - 1) % 4 == 0
        # 即 num_frames % 4 == 1
        adjusted_frame_count = target_frame_count
        if (adjusted_frame_count - 1) % 4 != 0:
            adjusted_frame_count = ((adjusted_frame_count - 1) // 4) * 4 + 1
        
        if adjusted_frame_count > target_frame_count:
            adjusted_frame_count -= 4
        
        logger.info(f"Video: {video_path}")
        logger.info(f"  Original: {total_frames} frames @ {orig_fps:.2f}fps")
        logger.info(f"  Target: {adjusted_frame_count} frames @ {actual_fps:.2f}fps")
        
        # 采样所有帧
        all_frames = []
        all_frame_ids = []
        
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, adjusted_frame_count, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((self.target_width, self.target_height), Image.BICUBIC)
            
            all_frames.append(frame_pil)
            all_frame_ids.append(int(idx))
        
        cap.release()
        
        metadata = {
            'orig_fps': orig_fps,
            'target_fps': actual_fps,
            'orig_frames': total_frames,
            'total_sampled_frames': len(all_frames),
            'height': self.target_height,
            'width': self.target_width
        }
        
        return all_frames, all_frame_ids, metadata
    
    def split_into_chunks(self, frames, frame_ids):
        """
        将帧序列分成多个chunk，每个chunk包含self.chunk_size帧
        
        Args:
            frames: list of PIL Images
            frame_ids: list of frame indices
        
        Returns:
            chunks: list of (chunk_frames, chunk_frame_ids, start_idx, end_idx)
        """
        chunks = []
        total_frames = len(frames)
        
        for start_idx in range(0, total_frames, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_frames)
            chunk_frames = frames[start_idx:end_idx]
            chunk_frame_ids = frame_ids[start_idx:end_idx]
            
            # 检查每个chunk的帧数是否满足约束
            chunk_len = len(chunk_frames)
            if (chunk_len - 1) % 4 != 0:
                # 如果不满足，调整到满足条件的帧数
                adjusted_len = ((chunk_len - 1) // 4) * 4 + 1
                chunk_frames = chunk_frames[:adjusted_len]
                chunk_frame_ids = chunk_frame_ids[:adjusted_len]
                logger.info(f"  Adjusted chunk from {chunk_len} to {adjusted_len} frames to satisfy (n-1)%4==0")
            
            chunks.append((chunk_frames, chunk_frame_ids, start_idx, end_idx))
        
        return chunks
    
    def frames_to_tensor(self, frames):
        """
        将PIL图像列表转换为归一化的视频tensor
        
        Args:
            frames: list of PIL Images
            
        Returns:
            tensor: [3, F, H, W] 归一化到[-1, 1]
        """
        tensors = []
        for frame in frames:
            # 转换为tensor并归一化到[-1, 1]
            tensor = TF.to_tensor(frame).sub_(0.5).div_(0.5)
            tensors.append(tensor)
        
        # [F, 3, H, W] -> [3, F, H, W]
        video_tensor = torch.stack(tensors, dim=1)
        return video_tensor


def extract_text_embeddings(text, tokenizer, text_encoder, device, num_videos_per_prompt=1, max_length=512, dtype=torch.float32):
    """提取文本的T5 embeddings"""
    if not text or text == "":
        return None
    
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    
    text = [text] if isinstance(text, str) else text
    text = [prompt_clean(u) for u in text]
    batch_size = len(text)
    
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        mask.to(device)
    ).last_hidden_state

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack([
        torch.cat(
            [u, u.new_zeros(max_length - u.size(0), u.size(1))])
        for u in prompt_embeds
    ],
                                dim=0)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                        seq_len, -1)
    
    return prompt_embeds.cpu()


def load_metadata(meta_path):
    """
    加载episodes.jsonl文件
    
    Args:
        meta_path: path to episodes.jsonl
    
    Returns:
        dict: mapping from episode_id to metadata
    """
    metadata = {}
    
    if not os.path.exists(meta_path):
        logger.warning(f"Metadata file not found: {meta_path}")
        return metadata
    
    with open(meta_path, 'r') as f:
        for line in f:
            try:
                episode_meta = json.loads(line.strip())
                # 假设jsonl中每条记录包含episode_id和text等信息
                episode_index = episode_meta.get('episode_index')
                if episode_index is not None:
                    metadata[str(episode_index)] = episode_meta
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line: {line}")
    
    logger.info(f"Loaded metadata for {len(metadata)} episodes")
    return metadata


def process_episode(video_processor, vae_encoder, tokenizer, text_encoder,
                    video_path, text, episode_id, save_dir, device, chunk_size=450):
    """
    处理单个episode：采样帧 -> 分块 -> 编码 -> 保存多个chunk文件
    
    Args:
        video_processor: VideoProcessor实例
        vae_encoder: WanVAEEncoder实例
        tokenizer: T5 tokenizer
        text_encoder: T5 text encoder
        video_path: 视频文件路径
        text: 文本指令
        episode_id: episode ID (用于文件名)
        save_dir: 保存目录
        device: torch device
        chunk_size: 每个chunk的帧数
    """
    # Step 1: 采样所有帧
    all_frames, all_frame_ids, metadata = video_processor.process_video(video_path)
    
    if len(all_frames) == 0:
        logger.error(f"No frames sampled from {video_path}")
        return
    
    # Step 2: 分成多个chunk
    video_processor.chunk_size = chunk_size
    chunks = video_processor.split_into_chunks(all_frames, all_frame_ids)
    
    # Step 3: 提取text embeddings (只需一次)
    text_emb = None
    if text:
        with torch.no_grad():
            text_emb = extract_text_embeddings(text, tokenizer, text_encoder, device)
    
    # Step 4: 处理每个chunk
    for chunk_idx, (chunk_frames, chunk_frame_ids, start_idx, end_idx) in enumerate(chunks):
        if len(chunk_frames) == 0:
            continue
        
        # 转换为tensor
        video_tensor = video_processor.frames_to_tensor(chunk_frames)  # [3, F, H, W]
        
        # VAE编码
        latent, C, F_lat, H_lat, W_lat = vae_encoder.encode_video(video_tensor)  # [C, F_lat, H_lat, W_lat]
        
        # 准备保存的数据
        latent_num_frames = F_lat 
        
        # 计算全局帧ID范围
        global_start_frame = chunk_frame_ids[0]
        global_end_frame = chunk_frame_ids[-1] + 1
        
        save_data = {
            'latent': latent,  #[7936, 48]
            'latent_num_frames': latent_num_frames, #31
            'latent_height': H_lat, #16
            'latent_width': W_lat, #16
            'video_num_frames': len(chunk_frames), #121
            'video_height': metadata['height'], #256
            'video_width': metadata['width'], #256
            'text_emb': text_emb.squeeze(0), #[512, 4096]
            'text': text,
            'frame_ids': chunk_frame_ids, # (121)
            'start_frame': global_start_frame, #0
            'end_frame': global_end_frame, #245
            'fps': metadata['target_fps'], 
            'ori_fps': metadata['orig_fps'],
        }
        
        # 保存文件: episode_{index}_{start_frame}_{end_frame}.pth
        save_filename = f'episode_{episode_id:06d}_{global_start_frame}_{global_end_frame}.pth'
        save_path = save_dir / save_filename
        torch.save(save_data, save_path)
        
        logger.info(f"Saved {save_path}")
        logger.info(f"  Video frames: {len(chunk_frames)} -> Latent frames: {latent_num_frames}")
        logger.info(f"  Frame range: {global_start_frame} - {global_end_frame}")
    
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Extract latents using Wan2.2 VAE")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin',
        help="config name from VA_CONFIGS"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of your_dataset (containing videos/, latents/, meta/)"
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        nargs='+',
        default=['cam_high'],
        help="Camera keys to process (e.g., cam_high, cam_left_wrist, cam_right_wrist)"
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10,
        help="Target FPS for frame sampling"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Target height (must be multiple of 32)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Target width (must be multiple of 32)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help="Number of frames per output pth file"
    )
    parser.add_argument(
        "--chunk-ids",
        type=str,
        nargs='+',
        help="Specific chunk IDs to process (e.g., 000 001). If not specified, process all."
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    init_logger()
    
    # 验证尺寸
    assert args.height % 32 == 0, f"Height must be multiple of 32, got {args.height}"
    assert args.width % 32 == 0, f"Width must be multiple of 32, got {args.width}"
    
    # 获取配置
    config = VA_CONFIGS[args.config_name]
    config.local_rank = args.local_rank
    device = torch.device(f"cuda:{args.local_rank}")
    
    # 初始化组件
    logger.info("Initializing VAE encoder...")
    vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'vae')
    vae_encoder = WanVAEEncoder(vae_path, device)
    
    logger.info("Loading tokenizer and text encoder...")
    tokenizer = load_tokenizer(
        os.path.join(config.wan22_pretrained_model_name_or_path, 'tokenizer')
    )
    text_encoder = load_text_encoder(
        os.path.join(config.wan22_pretrained_model_name_or_path, 'text_encoder'),
        torch_dtype=config.param_dtype,
        torch_device=device,
    )
    
    # 初始化视频处理器
    video_processor = VideoProcessor(
        target_fps=args.target_fps,
        target_height=args.height,
        target_width=args.width,
        chunk_size=args.chunk_size
    )
    
    # 设置目录
    dataset_root = Path(args.dataset_root)
    videos_dir = dataset_root / 'videos'
    latents_dir = dataset_root / 'latents'
    meta_dir = dataset_root / 'meta'
    
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return
    
    # 创建latents目录
    latents_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载metadata
    meta_file = meta_dir / 'episodes.jsonl'
    episode_metadata = load_metadata(meta_file)
    
    # 获取所有chunk目录
    if args.chunk_ids:
        chunk_dirs = [videos_dir / f'chunk-{chunk_id}' for chunk_id in args.chunk_ids]
        chunk_dirs = [d for d in chunk_dirs if d.exists()]
    else:
        chunk_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith('chunk-')])
    
    logger.info(f"Found {len(chunk_dirs)} chunks to process")
    
    total_episodes = 0
    total_chunks = 0
    
    for chunk_dir in chunk_dirs:
        chunk_id = chunk_dir.name.replace('chunk-', '')
        logger.info(f"Processing chunk {chunk_id}")
        
        for camera_key in args.camera_keys:
            # 视频目录: videos/chunk-{chunk_id}/observation.images.{camera_key}/
            video_camera_dir = chunk_dir / f'observation.images.{camera_key}'
            
            if not video_camera_dir.exists():
                logger.warning(f"Camera directory not found: {video_camera_dir}")
                continue
            
            # 查找所有mp4文件
            mp4_files = sorted(video_camera_dir.glob('episode_*.mp4'))
            logger.info(f"Found {len(mp4_files)} mp4 files in {video_camera_dir}")
            
            for mp4_file in mp4_files:
                # 从文件名提取episode_id
                # 文件名格式: episode_000000.mp4
                episode_name = mp4_file.stem  # episode_000000
                episode_id = int(episode_name.replace('episode_', ''))
                
                # 从metadata获取text
                text = ""
                if str(episode_id) in episode_metadata:
                    meta = episode_metadata[str(episode_id)]
                    text = meta.get('tasks', '')
                else:
                    logger.warning(f"No metadata found for episode {episode_id}")
                
                # 创建保存目录: latents/chunk-{chunk_id}/observation.images.{camera_key}/
                save_camera_dir = latents_dir / f'chunk-{chunk_id}' / f'observation.images.{camera_key}'
                save_camera_dir.mkdir(parents=True, exist_ok=True)
                
                # 处理这个episode
                num_chunks = process_episode(
                    video_processor=video_processor,
                    vae_encoder=vae_encoder,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    video_path=str(mp4_file),
                    text=text,
                    episode_id=episode_id,
                    save_dir=save_camera_dir,
                    device=device,
                    chunk_size=args.chunk_size
                )
                
                if num_chunks:
                    total_episodes += 1
                    total_chunks += num_chunks
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
    
    logger.info(f"Latent extraction completed! Processed {total_episodes} episodes, {total_chunks} chunks")


if __name__ == "__main__":
    main()
