# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_aloha_base_config = EasyDict(__name__='Config: RC ALOHA base')
rc_aloha_base_config.update(va_shared_cfg)

rc_aloha_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: These are for inference
rc_aloha_base_config.attn_window = 72
rc_aloha_base_config.frame_chunk_size = 2

rc_aloha_base_config.env_type = 'none'

rc_aloha_base_config.height = 256
rc_aloha_base_config.width = 320
rc_aloha_base_config.action_dim = 30
rc_aloha_base_config.action_per_frame = 16
rc_aloha_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.left_wrist',
    'observation.images.right_wrist'
]
rc_aloha_base_config.guidance_scale = 5
rc_aloha_base_config.action_guidance_scale = 1

rc_aloha_base_config.num_inference_steps = 25
rc_aloha_base_config.video_exec_step = -1
rc_aloha_base_config.action_num_inference_steps = 50

rc_aloha_base_config.snr_shift = 5.0
rc_aloha_base_config.action_snr_shift = 1.0


# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_aloha_base_config.used_action_channel_ids = list(range(0, 13)) + list(range(14, 19)) + list(range(21, 26)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(rc_aloha_base_config.used_action_channel_ids)
] * rc_aloha_base_config.action_dim
for i, j in enumerate(rc_aloha_base_config.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
rc_aloha_base_config.inverse_used_action_channel_ids = inverse_used_action_channel_ids



rc_aloha_pencil_case_config_train = EasyDict(__name__='Config: RC ALOHA pencil case')
rc_aloha_pencil_case_config_train.update(rc_aloha_base_config)
rc_aloha_pencil_case_config_train.action_norm_method = 'quantiles'
rc_aloha_pencil_case_config_train.norm_stat={
    "q01": [
  6.0679424e-01, -2.2940049e-01, -9.8167383e-04,  2.0655010e-02,
 -3.4262955e-03,  1.4843845e-01, -5.4128927e-01,  6.0194373e-01,
 -2.8486153e-02,  1.7661544e-02, -4.3897825e-01, -5.7565200e-04,
 -1.3499714e+00, -9.2118275e-01, -1.3456075e-01, -1.9396822e-01,
 -6.7752495e-02, -4.3610000e-04, -1.5347406e+00, -7.7800237e-02,
 -1.5706578e-01, -1.9508148e+00, -7.9999998e-04, -3.4000000e-03],
    "q99": [0.34181964, 0.00465323, 0.42854595, 0.72929543, 0.87002546, 0.12560092,
 0.7821621, 0.34912562, 0.32544464, 0.431879, 0.19072197, 0.98466605,
 0.5185191, 0.74653405, 0.20242018, 1.8691595, 0.00655894, 0.16404338,
 1.1954024, 1.844075, 0.7347064, 2.1204753, 0.00905344, 1.7361141,
 1.1046064, 0.29474035, 0.0599, 0.0677]
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
rc_aloha_pencil_case_config_train.batch_size = 2
rc_aloha_pencil_case_config_train.gradient_accumulation_steps = 4  # effective batch size = 4*8=32
rc_aloha_pencil_case_config_train.num_steps = 10000 
rc_aloha_pencil_case_config_train.save_root = "./checkpoints/rc_aloha_pencil_case/bs32lr2.5e-5"