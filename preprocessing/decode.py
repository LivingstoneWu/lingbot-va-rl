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

from diffusers.utils import export_to_video

class WanVAEEncoder:
    """Wan2.2 VAE Encoder - 用于批量编码视频帧"""

    def __init__(self, vae_path, device):
        self.device = device
        self.vae = load_vae(vae_path, torch_dtype=torch.float32, torch_device=device)
        self.vae_stride = (4, 8, 8)

        # 加载VAE的均值和标准差
        self.latents_mean = torch.tensor(self.vae.config.latents_mean)
        self.latents_std = torch.tensor(self.vae.config.latents_std)

        # 初始化video_processor
        from diffusers.video_processor import VideoProcessor
        self.video_processor = VideoProcessor(vae_scale_factor=1)

    def _reset_vae_cache(self):
        """彻底重置VAE的所有内部缓存"""
        # 清理VAE主对象的缓存
        if hasattr(self.vae, '_feat_map'):
            self.vae._feat_map = None
        if hasattr(self.vae, '_conv_idx'):
            self.vae._conv_idx = None
        if hasattr(self.vae, 'clear_cache'):
            self.vae.clear_cache()

        # 清理decoder的缓存
        if hasattr(self.vae, 'decoder'):
            if hasattr(self.vae.decoder, '_feat_map'):
                self.vae.decoder._feat_map = None
            if hasattr(self.vae.decoder, '_conv_idx'):
                self.vae.decoder._conv_idx = None
            if hasattr(self.vae.decoder, 'clear_cache'):
                self.vae.decoder.clear_cache()

        # 清理encoder的缓存（如果有）
        if hasattr(self.vae, 'encoder'):
            if hasattr(self.vae.encoder, '_feat_map'):
                self.vae.encoder._feat_map = None
            if hasattr(self.vae.encoder, '_conv_idx'):
                self.vae.encoder._conv_idx = None
            if hasattr(self.vae.encoder, 'clear_cache'):
                self.vae.encoder.clear_cache()

        # 强制清理GPU缓存
        torch.cuda.empty_cache()

    def decode_one_video(self, latents, F_lat=None, H_lat=None, W_lat=None, output_type='np'):
        """
        解码单个视频的latent，每次解码前重置缓存
        """
        # 在解码前彻底清理缓存
        self._reset_vae_cache()

        # 确保latents在正确的device和dtype
        latents = latents.to(self.device).to(self.vae.dtype)

        # 处理不同的输入格式
        if latents.dim() == 2:
            if F_lat is None or H_lat is None or W_lat is None:
                raise ValueError(
                    "When latents is in flatten format [F*H*W, C], "
                    "please provide F_lat, H_lat, and W_lat parameters"
                )

            C = latents.shape[1]
            expected_size = F_lat * H_lat * W_lat
            if latents.shape[0] != expected_size:
                raise ValueError(
                    f"Expected {expected_size} tokens, got {latents.shape[0]}. "
                    f"F_lat={F_lat}, H_lat={H_lat}, W_lat={W_lat}"
                )

            # 还原为 [C, F_lat, H_lat, W_lat] 格式
            latents = latents.view(F_lat, H_lat, W_lat, C).permute(3, 0, 1, 2).contiguous()

        elif latents.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")

        # 添加batch维度 [1, C, F, H, W]
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)

        # 反归一化
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean

        # VAE解码
        video = self.vae.decode(latents, return_dict=False)[0]



        # 确保video没有梯度，然后后处理
        video = video.detach()  # 添加这一行，detach梯度

        # 后处理
            # 后处理
        if output_type == 'np':
            # 转换为 [1, F, H, W, C] 格式
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            # 对于 pt 格式，保持 [1, C, F, H, W]
            # 但需要反归一化到 [0, 1]
            video = (video + 1) / 2
            video = video.clamp(0, 1)

        # 解码后再次清理缓存
        self._reset_vae_cache()

        return video

    def decode_in_parts(self, latents, F_lat, H_lat, W_lat, chunk_size=8, output_type='np'):
        """
        沿时间维度分块解码latent，每块chunk_size帧
        """
        # 确保latents在正确的device
        latents = latents.to(self.device).to(self.vae.dtype)
        C = latents.shape[1]

        # 验证维度
        expected_size = F_lat * H_lat * W_lat
        if latents.shape[0] != expected_size:
            raise ValueError(
                f"Expected {expected_size} tokens, got {latents.shape[0]}. "
                f"F_lat={F_lat}, H_lat={H_lat}, W_lat={W_lat}"
            )

        # 恢复为 [F_lat, H_lat, W_lat, C] 格式
        latent_reshaped = latents.view(F_lat, H_lat, W_lat, C)

        decoded_chunks = []

        # 沿时间维度分块
        for start_f in range(0, F_lat, chunk_size):
            end_f = min(start_f + chunk_size, F_lat)
            actual_chunk_size = end_f - start_f

            print(f"Decoding chunk {start_f}-{end_f} ({actual_chunk_size} latent frames)")

            # 提取当前块的latent
            chunk_latent = latent_reshaped[start_f:end_f]  # [chunk, H, W, C]

            # 展平为 [chunk * H * W, C]
            chunk_flat = chunk_latent.reshape(-1, C)

            # 解码这一块，先得到 tensor 格式
            chunk_video = self.decode_one_video(
                chunk_flat,
                actual_chunk_size,
                H_lat,
                W_lat,
                output_type='pt'
            )

            # chunk_video 应该是 [1, 3, video_frames, H, W]
            video_frames = chunk_video.shape[2]
            print(f"  Decoded {video_frames} video frames, shape: {chunk_video.shape}")

            # 收集解码结果
            decoded_chunks.append(chunk_video)

            # 每解码完一块，强制清理
            torch.cuda.empty_cache()

        # 沿着时间维度（维度2）拼接所有chunk
        # 所有chunk应该都有相同的 [1, 3, ?, H, W] 格式，只是第三维（帧数）不同
        video = torch.cat(decoded_chunks, dim=2)  # [1, 3, total_frames, H, W]

        total_frames = video.shape[2]
        print(f"After concatenation: {video.shape} (total {total_frames} video frames)")

        # 转换为numpy如果需要
        if output_type == 'np':
            # 转换为 [F, H, W, C] 格式
            video = video.squeeze(0)  # [3, total_frames, H, W]
            video = video.permute(1, 2, 3, 0).cpu().numpy()  # [total_frames, H, W, 3]

            print("####video max", video.max())

        print(f"Decoding completed: total {video.shape[1] if output_type=='np' else video.shape[2]} video frames")
        print(f"Final video shape: {video.shape}")

        return video


    def decode_on_cpu(self, latents, F_lat, H_lat, W_lat, output_type='np', chunk_size=8):
        """
        在CPU上解码，彻底避免GPU显存问题
        """
        print("Using CPU decoding...")

        # 保存原始设备
        original_device = next(self.vae.parameters()).device

        # 将VAE移到CPU
        self.vae = self.vae.to('cpu')

        # 确保latents也在CPU上
        if latents.is_cuda:
            latents = latents.cpu()

        try:
            # 在CPU上分块解码
            video = self.decode_in_parts(latents, F_lat, H_lat, W_lat, chunk_size=chunk_size, output_type=output_type)
        finally:
            # 恢复VAE到原设备
            self.vae = self.vae.to(original_device)
            torch.cuda.empty_cache()

        return video


def decode_and_save(vae_encoder, pth_path, save_path, use_cpu=False, chunk_size=8, fps=10):
    """
    解码并保存视频
    """
    # 加载latent数据
    data = torch.load(pth_path, map_location='cpu', weights_only=False)
    latent = data['latent']
    F_lat = data['latent_num_frames']
    H_lat = data['latent_height']
    W_lat = data['latent_width']
    
    print(f"Loading latent: {latent.shape}, F_lat={F_lat}, H_lat={H_lat}, W_lat={W_lat}")
    
    # 选择解码方式
    if use_cpu:
        decoded_video = vae_encoder.decode_on_cpu(latent, F_lat, H_lat, W_lat, output_type='np', chunk_size=chunk_size)
    else:
        decoded_video = vae_encoder.decode_in_parts(
            latent, F_lat, H_lat, W_lat, chunk_size=chunk_size, output_type='np'
        )
    
    # decoded_video 形状应该是 [1, F, H, W, C] 或 [F, H, W, C]
    print(f"Decoded video shape before processing: {decoded_video.shape}")
    
    # 去掉 batch 维度（如果是5维）
    if len(decoded_video.shape) == 5:
        video_to_save = decoded_video[0]  # [F, H, W, C]
    else:
        video_to_save = decoded_video
    

    print(f"Video to save shape: {video_to_save.shape}")
    
    # 验证通道数
    if video_to_save.shape[-1] not in [1, 2, 3, 4]:
        raise ValueError(f"Invalid number of channels: {video_to_save.shape[-1]}")
    

    export_to_video(video_to_save, save_path, fps=fps)
    print(f"Video saved to {save_path}")
    
    return decoded_video

# 使用示例
#def example_usage():
#    # 假设已经有 vae_encoder 和 latent_data
#    # vae_encoder = WanVAEEncoder(...)
#    # latent_data = torch.load('episode_000000_0_450.pth')['latent']

#    # 保存为视频
#    decode_and_save(
#        vae_encoder=vae_encoder,
#        latent_data=latent_data,
#        save_path='/path/to/output_video.mp4',
#        fps=10,
#        save_mode='video'
#    )

#    # 保存为帧图像
#    decode_and_save(
#        vae_encoder=vae_encoder,
#        latent_data=latent_data,
#        save_path='/path/to/frames_dir',
#        fps=10,
#        save_mode='frames'
#    )

if __name__ == "__main__":
    config = VA_CONFIGS["demo"]
    config.local_rank = '0' 
    device = torch.device(f"cuda:0")
    logger.info("Initializing VAE encoder...")
    vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'vae')
    vae_encoder= WanVAEEncoder(vae_path, device)

    logger.info("Loading tokenizer and text encoder...")
    tokenizer = load_tokenizer(
        os.path.join(config.wan22_pretrained_model_name_or_path, 'tokenizer')
    )
    text_encoder = load_text_encoder(
        os.path.join(config.wan22_pretrained_model_name_or_path, 'text_encoder'),
        torch_dtype=config.param_dtype,
        torch_device=device,
    )

    # 读取latents.pth
    #pth_path ='/liujinxin/code/lhc/lingbot-va/pick-n-place-sq-lerobot-v21/latents/chunk-000/observation.images.top/episode_000000_0_264.pth'
    #pth_path ='/liujinxin/code/lhc/lingbot-va/pick-n-place-sq-lerobot-v21/latents_generate/chunk-000/observation.images.top/episode_000000_000000_000264.pth'
    pth_path = '/liujinxin/code/lhc/lingbot-va/datasets/robochallenge/turn_on_faucet_trim/latents/chunk-000/observation.images.top/episode_000000_0_518.pth'


    decode_and_save(
        vae_encoder=vae_encoder,
        pth_path=pth_path,
        save_path='./output_video.mp4',
        fps=15,
        )


