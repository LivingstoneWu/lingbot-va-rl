# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_ur5_base_config = EasyDict(__name__='Config: RC UR5 arm base')
rc_ur5_base_config.update(va_shared_cfg)

rc_ur5_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: These are for inference
rc_ur5_base_config.attn_window = 72
rc_ur5_base_config.frame_chunk_size = 2

rc_ur5_base_config.env_type = 'none'

rc_ur5_base_config.height = 256
rc_ur5_base_config.width = 320
rc_ur5_base_config.action_dim = 30
rc_ur5_base_config.action_per_frame = 16
rc_ur5_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.wrist'
]
rc_ur5_base_config.guidance_scale = 5
rc_ur5_base_config.action_guidance_scale = 1

rc_ur5_base_config.num_inference_steps = 25
rc_ur5_base_config.video_exec_step = -1
rc_ur5_base_config.action_num_inference_steps = 50

rc_ur5_base_config.snr_shift = 5.0
rc_ur5_base_config.action_snr_shift = 1.0


# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_ur5_base_config.used_action_channel_ids = list(range(0, 7)) + list(range(14, 20)) + [28]
inverse_used_action_channel_ids = [
    len(rc_ur5_base_config.used_action_channel_ids)
] * rc_ur5_base_config.action_dim
for i, j in enumerate(rc_ur5_base_config.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
rc_ur5_base_config.inverse_used_action_channel_ids = inverse_used_action_channel_ids



rc_single_set_the_plates_config_train = EasyDict(__name__='Config: RC UR5 set the plates')
rc_single_set_the_plates_config_train.update(rc_ur5_base_config)
rc_single_set_the_plates_config_train.action_norm_method = 'quantiles'
rc_single_set_the_plates_config_train.norm_stat = {
    "q01": [-0.70612377, -0.25594482, 0.24335368, -0.65606493, -0.83061516, -0.67004627, -0.47191674, 0, 0, 0, 0, 0, 0, 0, -0.56730443, -2.1418378, -2.0096891, -1.571, 0.62312514, -1.7936109, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    "q99": [-0.41302848, 0.36700642, 0.59971273, 0.98459029, 0.98832572, 0.69834197, 0.69406056, 0, 0, 0, 0, 0, 0, 0, 0.77211893, -1.4448384, -1.12301, 0.41157055, 2.0893166, 1.8927952, 0, 0, 0, 0, 0, 0, 0, 0, 226, 0],
}

rc_single_set_the_plates_config_train.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/set_the_plates'
rc_single_set_the_plates_config_train.empty_emb_path = os.path.join(rc_single_set_the_plates_config_train.dataset_path, 'empty_emb.pt')
rc_single_set_the_plates_config_train.enable_wandb = False 
rc_single_set_the_plates_config_train.load_worker = 2
rc_single_set_the_plates_config_train.save_interval = 2000
rc_single_set_the_plates_config_train.gc_interval = 50
rc_single_set_the_plates_config_train.cfg_prob = 0.1

# Training parameters
rc_single_set_the_plates_config_train.learning_rate = 2.5e-5
rc_single_set_the_plates_config_train.beta1 = 0.9
rc_single_set_the_plates_config_train.beta2 = 0.95
rc_single_set_the_plates_config_train.weight_decay = 1e-1
rc_single_set_the_plates_config_train.warmup_steps = 50
rc_single_set_the_plates_config_train.min_lr = 1e-6
rc_single_set_the_plates_config_train.batch_size = 1
rc_single_set_the_plates_config_train.gradient_accumulation_steps = 2  # effective batch size = 2*8=16
rc_single_set_the_plates_config_train.num_steps = 10000 
rc_single_set_the_plates_config_train.save_root = "./checkpoints/rc_set_the_plates/bs16lr2.5e-5"