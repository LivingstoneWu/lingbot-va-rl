# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_franka_base_config = EasyDict(__name__='Config: RC UR5 arm base')
rc_franka_base_config.update(va_shared_cfg)

rc_franka_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: the latent chunk size, extraction 501 chunk_size -> (501-1)/4 + 1 = 126 frames
rc_franka_base_config.max_latent_frames = 126

# COMMENT: These are for inference
rc_franka_base_config.attn_window = 72
rc_franka_base_config.frame_chunk_size = 4  # how many latent frames are generated per inference
rc_franka_base_config.num_inference_steps = 10
rc_franka_base_config.video_exec_step = -1
rc_franka_base_config.action_num_inference_steps = 20

rc_franka_base_config.env_type = 'none'

rc_franka_base_config.height = 256
rc_franka_base_config.width = 256
rc_franka_base_config.action_dim = 30
rc_franka_base_config.action_per_frame = 12
rc_franka_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.wrist', 'observation.images.scene'
]
rc_franka_base_config.guidance_scale = 5
rc_franka_base_config.action_guidance_scale = 1

rc_franka_base_config.snr_shift = 5.0
rc_franka_base_config.action_snr_shift = 1.0
rc_franka_base_config.infer_mode = 'server'
rc_franka_base_config.save_root = './inf_out'

def inverse_ids(cfg):
    inverse_used_action_channel_ids = [
        len(cfg.used_action_channel_ids)
    ] * cfg.action_dim
    for i, j in enumerate(cfg.used_action_channel_ids):
        inverse_used_action_channel_ids[j] = i
    return inverse_used_action_channel_ids

# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_franka_base_config.used_action_channel_ids = list(range(14, 21)) + [28]

rc_franka_base_config.inverse_used_action_channel_ids = inverse_ids(rc_franka_base_config)

rc_franka_press_button = EasyDict(__name__='Config: RC UR5 set the plates')
rc_franka_press_button.update(rc_franka_base_config)
rc_franka_press_button.action_norm_method = 'quantiles'
rc_franka_press_button.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.21551314, -0.78463328, -0.28834566, -2.7551956, -0.10172039, 1.962543, -0.30158946, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21330975, 0.29195952, 0.25587457, -1.9371207, 0.089850545, 2.5238667, 0.33904356, 0, 0, 0, 0, 0, 0, 0, 0.085000001, 0],
}


rc_franka_press_button.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/press_three_buttons'
rc_franka_press_button.empty_emb_path = os.path.join(rc_franka_press_button.dataset_path, 'empty_emb.pt')
rc_franka_press_button.enable_wandb = False 
rc_franka_press_button.load_worker = 1
rc_franka_press_button.save_interval = 500
rc_franka_press_button.gc_interval = 50
rc_franka_press_button.cfg_prob = 0.1
rc_franka_press_button.inverse_used_action_channel_ids = inverse_ids(rc_franka_press_button)

# Training parameters
rc_franka_press_button.learning_rate = 2.5e-5
rc_franka_press_button.beta1 = 0.9
rc_franka_press_button.beta2 = 0.95
rc_franka_press_button.weight_decay = 1e-1
rc_franka_press_button.warmup_steps = 50
rc_franka_press_button.min_lr = 1e-6
rc_franka_press_button.batch_size = 1
rc_franka_press_button.gradient_accumulation_steps = 1  # effective batch size = 2*8=16
rc_franka_press_button.num_steps = 15000
rc_franka_press_button.save_root = "./checkpoints/rc_press_buttons/bs4lr2.5e-5"