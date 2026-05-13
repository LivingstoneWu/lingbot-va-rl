# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_ur5_base_config = EasyDict(__name__='Config: RC UR5 arm base')
rc_ur5_base_config.update(va_shared_cfg)

rc_ur5_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: the latent chunk size, extraction 501 chunk_size -> (501-1)/4 + 1 = 126 frames
rc_ur5_base_config.max_latent_frames = 126

# COMMENT: These are for inference
rc_ur5_base_config.attn_window = 72
rc_ur5_base_config.frame_chunk_size = 4  # how many latent frames are generated per inference

rc_ur5_base_config.env_type = 'none'

rc_ur5_base_config.height = 256
rc_ur5_base_config.width = 256
rc_ur5_base_config.action_dim = 30
rc_ur5_base_config.action_per_frame = 12
rc_ur5_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.wrist'
]
rc_ur5_base_config.guidance_scale = 5
rc_ur5_base_config.action_guidance_scale = 1

rc_ur5_base_config.num_inference_steps = 10
rc_ur5_base_config.video_exec_step = -1
rc_ur5_base_config.action_num_inference_steps = 20

rc_ur5_base_config.snr_shift = 5.0
rc_ur5_base_config.action_snr_shift = 1.0
rc_ur5_base_config.infer_mode = 'server'
rc_ur5_base_config.save_root = './inf_out'

def inverse_ids(cfg):
    inverse_used_action_channel_ids = [
        len(cfg.used_action_channel_ids)
    ] * cfg.action_dim
    for i, j in enumerate(cfg.used_action_channel_ids):
        inverse_used_action_channel_ids[j] = i
    return inverse_used_action_channel_ids

# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_ur5_base_config.used_action_channel_ids = list(range(14, 20)) + [28]

rc_ur5_base_config.inverse_used_action_channel_ids = inverse_ids(rc_ur5_base_config)



rc_single_set_the_plates_config_train = EasyDict(__name__='Config: RC UR5 set the plates')
rc_single_set_the_plates_config_train.update(rc_ur5_base_config)
rc_single_set_the_plates_config_train.action_norm_method = 'quantiles'
rc_single_set_the_plates_config_train.used_action_channel_ids = list(range(14, 20)) + [28]
rc_single_set_the_plates_config_train.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.56730443, -2.1418378, -2.0096891, -1.571, 0.62312514, -1.7936109, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.77211893, -1.4448384, -1.12301, 0.41157055, 2.0893166, 1.8927952, 0, 0, 0, 0, 0, 0, 0, 0, 226, 0],
}

rc_single_set_the_plates_config_train.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/set_the_plates'
rc_single_set_the_plates_config_train.empty_emb_path = os.path.join(rc_single_set_the_plates_config_train.dataset_path, 'empty_emb.pt')
rc_single_set_the_plates_config_train.enable_wandb = False 
rc_single_set_the_plates_config_train.load_worker = 1
rc_single_set_the_plates_config_train.save_interval = 500
rc_single_set_the_plates_config_train.gc_interval = 50
rc_single_set_the_plates_config_train.cfg_prob = 0.1
rc_single_set_the_plates_config_train.inverse_used_action_channel_ids = inverse_ids(rc_single_set_the_plates_config_train)

# Training parameters
rc_single_set_the_plates_config_train.learning_rate = 2.5e-5
rc_single_set_the_plates_config_train.beta1 = 0.9
rc_single_set_the_plates_config_train.beta2 = 0.95
rc_single_set_the_plates_config_train.weight_decay = 1e-1
rc_single_set_the_plates_config_train.warmup_steps = 50
rc_single_set_the_plates_config_train.min_lr = 1e-6
rc_single_set_the_plates_config_train.batch_size = 1
rc_single_set_the_plates_config_train.gradient_accumulation_steps = 4  # effective batch size = 2*8=16
rc_single_set_the_plates_config_train.num_steps = 6000
rc_single_set_the_plates_config_train.save_root = "./checkpoints/rc_set_the_plates/bs16lr2.5e-5"


rc_single_stack_color_blocks_config_train = EasyDict(__name__='Config: RC UR5 stack color blocks')
rc_single_stack_color_blocks_config_train.update(rc_single_set_the_plates_config_train)
rc_single_stack_color_blocks_config_train.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.057282273, -2.2433136, -1.696568, -1.6897739, 1.5707246, -1.3038472, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.26362914, -1.5708085, -0.86829638, -1.1269208, 1.5708603, 1.4265021, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0],
}
rc_single_stack_color_blocks_config_train.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/stack_color_blocks'
rc_single_stack_color_blocks_config_train.empty_emb_path = os.path.join(rc_single_stack_color_blocks_config_train.dataset_path, 'empty_emb.pt')
rc_single_stack_color_blocks_config_train.save_root = "./checkpoints/rc_stack_color_blocks_perfectaligned/bs4lr2.5e-5"
rc_single_stack_color_blocks_config_train.batch_size = 1
rc_single_stack_color_blocks_config_train.gradient_accumulation_steps = 1
rc_single_stack_color_blocks_config_train.num_steps = 15000
# COMMENT: inference sppedup
rc_single_stack_color_blocks_config_train.video_exec_step = 10



rc_single_arrange_fruits_config_train = EasyDict(__name__='Config: RC UR5 arrange fruits')
rc_single_arrange_fruits_config_train.update(rc_single_set_the_plates_config_train)
rc_single_arrange_fruits_config_train.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.61907941, -2.2372825, -2.004142, -1.7560807, 1.5706294, -1.6674976, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.71614951, -1.4155082, -0.79284388, -1.0610365, 1.5709291, 1.6951444, 0, 0, 0, 0, 0, 0, 0, 0, 154, 0],
}
rc_single_arrange_fruits_config_train.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/arrange_fruits_in_basket'
rc_single_arrange_fruits_config_train.empty_emb_path = os.path.join(rc_single_arrange_fruits_config_train.dataset_path, 'empty_emb.pt')
rc_single_arrange_fruits_config_train.save_root = "./checkpoints/arrange_fruits_in_basket/bs4lr2.5e-5"
rc_single_arrange_fruits_config_train.gradient_accumulation_steps = 1




local_ur5_stack_color_blocks_config = EasyDict(__name__='Config: RC UR5 stack color blocks')
local_ur5_stack_color_blocks_config.update(rc_single_set_the_plates_config_train)
# COMMENT: the local ur5 dataset fps is 10! i.e. not further downsampled!
local_ur5_stack_color_blocks_config.action_per_frame = 4 
local_ur5_stack_color_blocks_config.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3103613, -1.7462763, 1.3626062, -2.6289268, -1.8888395, -1.1673909, 0, 0, 0, 0, 0, 0, 0, 0, 0.011764706, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.77189082, -0.77308255, 2.2129431, -1.9355265, -1.4393553, 0.55144018, 0, 0, 0, 0, 0, 0, 0, 0, 0.53333336, 0],
}
local_ur5_stack_color_blocks_config.dataset_path = '//liujinxin/code/lhc/wy/wms/lingbot-va/datasets/ur5e/stack_color_blocks_action_corrected'
local_ur5_stack_color_blocks_config.empty_emb_path = os.path.join(local_ur5_stack_color_blocks_config.dataset_path, 'empty_emb.pt')
local_ur5_stack_color_blocks_config.save_root = "./checkpoints/local_stack_color_blocks/bs8lr2.5e-5"
local_ur5_stack_color_blocks_config.batch_size = 2
# local_ur5_stack_color_blocks_config.gradient_accumulation_steps = 2

# COMMENT: for inference
local_ur5_stack_color_blocks_config.video_exec_step = 10



rc_ur5_shred_papers = EasyDict(__name__='Config: RC UR5 stack color blocks')
rc_ur5_shred_papers.update(rc_single_set_the_plates_config_train)
rc_ur5_shred_papers.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/shred_scrap_paper'
rc_ur5_shred_papers.empty_emb_path = os.path.join(rc_ur5_shred_papers.dataset_path, 'empty_emb.pt')
rc_ur5_shred_papers.save_root = "./checkpoints/shred_scrap_papers/bs8lr2.5e-5"
rc_ur5_shred_papers.gradient_accumulation_steps = 2
rc_ur5_shred_papers.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5338214, -2.1226137, -1.7966197, -2.1330581, 1.5706866, -1.8574237, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.72634566, -1.4174631, -0.48624778, -1.3429415, 1.5709201, 1.8614224, 0, 0, 0, 0, 0, 0, 0, 0, 228, 0],
}
