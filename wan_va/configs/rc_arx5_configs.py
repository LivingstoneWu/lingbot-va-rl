# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_arx5_base_config = EasyDict(__name__='Config: RC UR5 arm base')
rc_arx5_base_config.update(va_shared_cfg)

rc_arx5_base_config.wan22_pretrained_model_name_or_path = "/luhongchao/shared/weights/lingbot-va-base"

# COMMENT: the latent chunk size, extraction 501 chunk_size -> (501-1)/4 + 1 = 126 frames
rc_arx5_base_config.max_latent_frames = 126

# COMMENT: These are for inference
rc_arx5_base_config.attn_window = 72
rc_arx5_base_config.frame_chunk_size = 4  # how many latent frames are generated per inference

rc_arx5_base_config.env_type = 'none'

rc_arx5_base_config.height = 256
rc_arx5_base_config.width = 256
rc_arx5_base_config.action_dim = 30
rc_arx5_base_config.action_per_frame = 12
rc_arx5_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.wrist', 'observation.images.scene'
]
rc_arx5_base_config.guidance_scale = 5
rc_arx5_base_config.action_guidance_scale = 1

rc_arx5_base_config.num_inference_steps = 20
rc_arx5_base_config.video_exec_step = 10
rc_arx5_base_config.action_num_inference_steps = 10

rc_arx5_base_config.snr_shift = 5.0
rc_arx5_base_config.action_snr_shift = 1.0
rc_arx5_base_config.infer_mode = 'server'
rc_arx5_base_config.save_root = './inf_out'

def inverse_ids(cfg):
    inverse_used_action_channel_ids = [
        len(cfg.used_action_channel_ids)
    ] * cfg.action_dim
    for i, j in enumerate(cfg.used_action_channel_ids):
        inverse_used_action_channel_ids[j] = i
    return inverse_used_action_channel_ids

# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_arx5_base_config.used_action_channel_ids =  list(range(14, 20)) + [28]
rc_arx5_base_config.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.1285954, 0.74826431, 0.28706074, -1.4375906, -1.026741, -1.5409708, 0, 0, 0, 0, 0, 0, 0, 0, 0.00039041872, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.107233, 2.2024488, 2.1372166, 1.3506145, 0.71507549, 1.7362862, 0, 0, 0, 0, 0, 0, 0, 0, 0.087184839, 0],
}

rc_arx5_base_config.inverse_used_action_channel_ids = inverse_ids(rc_arx5_base_config)

rc_arx5_base_config.dataset_path = '/luhongchao/shared/dataset/robochallenge_converted/robochallenge_v1/arx5'
rc_arx5_base_config.empty_emb_path = '/luhongchao/shared/dataset/robochallenge_converted/robochallenge_v1/arrange_flowers/empty_emb.pt'
rc_arx5_base_config.enable_wandb = False 
rc_arx5_base_config.load_worker = 1
rc_arx5_base_config.save_interval = 5000
rc_arx5_base_config.gc_interval = 50
rc_arx5_base_config.cfg_prob = 0.1
rc_arx5_base_config.learning_rate = 3e-5
rc_arx5_base_config.beta1 = 0.9
rc_arx5_base_config.beta2 = 0.95
rc_arx5_base_config.weight_decay = 1e-1
rc_arx5_base_config.warmup_steps = 50
rc_arx5_base_config.min_lr = 1e-6
rc_arx5_base_config.batch_size = 1
rc_arx5_base_config.gradient_accumulation_steps = 2  # effective batch size = 2*8=16
rc_arx5_base_config.num_steps = 30000
rc_arx5_base_config.save_root = "./checkpoints/rc_arx5_base_debug/bs16lr3e-5_1e-6"
rc_arx5_base_config.frame_chunk_size = 2




rc_arx5_arrange_flowers = EasyDict(__name__='Config: RC ARX5 set the plates')
rc_arx5_arrange_flowers.update(rc_arx5_base_config)
rc_arx5_arrange_flowers.action_norm_method = 'quantiles'
rc_arx5_arrange_flowers.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.96150875, 0.76390457, 0.33588886, -1.3723583, -0.83867359, -1.5356302, 0, 0, 0, 0, 0, 0, 0, 0, 0.0004705046, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.55256748, 2.175746, 1.882391, 1.5466928, 0.75818253, 0.47436523, 0, 0, 0, 0, 0, 0, 0, 0, 0.087171488, 0],
}


rc_arx5_arrange_flowers.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/arrange_flowers'
rc_arx5_arrange_flowers.empty_emb_path = os.path.join(rc_arx5_arrange_flowers.dataset_path, 'empty_emb.pt')
rc_arx5_arrange_flowers.enable_wandb = False 
rc_arx5_arrange_flowers.load_worker = 1
rc_arx5_arrange_flowers.save_interval = 500
rc_arx5_arrange_flowers.gc_interval = 50
rc_arx5_arrange_flowers.cfg_prob = 0.1
rc_arx5_arrange_flowers.inverse_used_action_channel_ids = inverse_ids(rc_arx5_arrange_flowers)

# Training parameters
rc_arx5_arrange_flowers.learning_rate = 2.5e-5
rc_arx5_arrange_flowers.beta1 = 0.9
rc_arx5_arrange_flowers.beta2 = 0.95
rc_arx5_arrange_flowers.weight_decay = 1e-1
rc_arx5_arrange_flowers.warmup_steps = 50
rc_arx5_arrange_flowers.min_lr = 1e-6
rc_arx5_arrange_flowers.batch_size = 1
rc_arx5_arrange_flowers.gradient_accumulation_steps = 2  # effective batch size = 2*8=16
rc_arx5_arrange_flowers.num_steps = 6000
rc_arx5_arrange_flowers.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"
# rc_arx5_arrange_flowers.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/rc_arrange_flowers/bs16lr2.5e-5/checkpoints/checkpoint_step_2000"
rc_arx5_arrange_flowers.save_root = "./checkpoints/rc_arrange_flowers/bs8lr2.5e-5_resume2000"
rc_arx5_arrange_flowers.frame_chunk_size = 2


rc_arx5_place_shoes = EasyDict(__name__='Config: RC ARX5 set the plates')
rc_arx5_place_shoes.update(rc_arx5_arrange_flowers)
rc_arx5_place_shoes.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.94167233, 0.75398636, 0.26951218, -1.4337759, -0.8360033, -0.82989979, 0, 0, 0, 0, 0, 0, 0, 0, 0.00040376637, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2087049, 2.0033188, 2.0204849, 0.85889244, 0.70019817, 0.78526783, 0, 0, 0, 0, 0, 0, 0, 0, 0.087184839, 0],
}
rc_arx5_place_shoes.save_root = './checkpoints/rc_place_shoes/bs8lr2.5e-5'
rc_arx5_place_shoes.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"
rc_arx5_place_shoes.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/place_shoes_on_rack'
rc_arx5_place_shoes.frame_chunk_size = 3

rc_arx5_wipe_table = EasyDict(__name__='Config: RC ARX5 set the plates')
rc_arx5_wipe_table.update(rc_arx5_arrange_flowers)
rc_arx5_wipe_table.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7856493, 0.76237869, 0.24052048, -1.4120321, -0.67730999, -1.2773705, 0, 0, 0, 0, 0, 0, 0, 0, 0.00013013957, 0],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0908289, 1.9911118, 1.9361792, 1.339551, 0.6696806, 1.5238037, 0, 0, 0, 0, 0, 0, 0, 0, 0.087184839, 0],
}
rc_arx5_wipe_table.save_root = './checkpoints/rc_wipe_table/bs8lr2.5e-5'
rc_arx5_wipe_table.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/wipe_the_table'