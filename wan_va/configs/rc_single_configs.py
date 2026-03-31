# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_single_base_config = EasyDict(__name__='Config: RC Single arm base')
rc_single_base_config.update(va_shared_cfg)

rc_single_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: These are for inference
rc_single_base_config.attn_window = 72
rc_single_base_config.frame_chunk_size = 2

rc_single_base_config.env_type = 'none'

rc_single_base_config.height = 256
rc_single_base_config.width = 320
rc_single_base_config.action_dim = 30
rc_single_base_config.action_per_frame = 16
rc_single_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.scene',
    'observation.images.wrist'
]
rc_single_base_config.guidance_scale = 5
rc_single_base_config.action_guidance_scale = 1

rc_single_base_config.num_inference_steps = 25
rc_single_base_config.video_exec_step = -1
rc_single_base_config.action_num_inference_steps = 50

rc_single_base_config.snr_shift = 5.0
rc_single_base_config.action_snr_shift = 1.0


# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_single_base_config.used_action_channel_ids = list(range(0, 7)) + list(range(14, 20)) + [28]
inverse_used_action_channel_ids = [
    len(rc_single_base_config.used_action_channel_ids)
] * rc_single_base_config.action_dim
for i, j in enumerate(rc_single_base_config.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
rc_single_base_config.inverse_used_action_channel_ids = inverse_used_action_channel_ids



rc_aloha_pencil_case_config_train = EasyDict(__name__='Config: RC ALOHA pencil case')
rc_aloha_pencil_case_config_train.update(rc_single_base_config)
rc_aloha_pencil_case_config_train.action_norm_method = 'quantiles'
rc_aloha_pencil_case_config_train.norm_stat = {
    "q01": [0.031551607, -0.14152533, 0.15114924, -0.083684184, 0.6086638, -0.23346399, 0.0018790263, 0.022730829, -0.0038374024, 0.1484917, -0.54672539, 0.60530323, -0.030355122, -0.15870552, 0.018053232, -0.43822816, -0.000627984, -1.3246624, -0.90175295, -0.15870552, -0.15870552, -0.1313882, -0.19809407, -0.072915919, -0.00045354399, -1.5296671, -0.15870552, -0.15870552, -0.15870552, -0.085161611],
    "q99": [0.33579153, 0.0044819559, 0.42525959, 0.72795093, 0.86931777, 0.11442605, 0.77959669, 0.34829536, 0.32478839, 0.42948574, 0.19060747, 0.9860695, 0.51719558, 1.1026527, 0.74680978, 0.17438766, 1.8653436, 0.0066287201, 0.15326571, 1.1026527, 1.1026527, 1.194129, 1.8375188, 0.73050237, 2.1186435, 0.0090185478, 1.1026527, 1.1026527, 1.1026527, 1.7354861],
}

rc_aloha_pencil_case_config_train.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case'
rc_aloha_pencil_case_config_train.empty_emb_path = os.path.join(rc_aloha_pencil_case_config_train.dataset_path, 'empty_emb.pt')
rc_aloha_pencil_case_config_train.enable_wandb = False 
rc_aloha_pencil_case_config_train.load_worker = 2
rc_aloha_pencil_case_config_train.save_interval = 2000
rc_aloha_pencil_case_config_train.gc_interval = 50
rc_aloha_pencil_case_config_train.cfg_prob = 0.1

# Training parameters
rc_aloha_pencil_case_config_train.learning_rate = 2.5e-5
rc_aloha_pencil_case_config_train.beta1 = 0.9
rc_aloha_pencil_case_config_train.beta2 = 0.95
rc_aloha_pencil_case_config_train.weight_decay = 1e-1
rc_aloha_pencil_case_config_train.warmup_steps = 50
rc_aloha_pencil_case_config_train.batch_size = 1
rc_aloha_pencil_case_config_train.gradient_accumulation_steps = 2  # effective batch size = 2*8=16
rc_aloha_pencil_case_config_train.num_steps = 10000 
rc_aloha_pencil_case_config_train.save_root = "./checkpoints/rc_aloha_pencil_case/bs32lr2.5e-5"