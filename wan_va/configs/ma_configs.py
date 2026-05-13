# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

def inverse_ids(cfg):
    inverse_used_action_channel_ids = [
        len(cfg.used_action_channel_ids)
    ] * cfg.action_dim
    for i, j in enumerate(cfg.used_action_channel_ids):
        inverse_used_action_channel_ids[j] = i
    return inverse_used_action_channel_ids

ma_base_config = EasyDict(__name__='Config: RC ALOHA base')
ma_base_config.update(va_shared_cfg)

ma_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: These are for inference
ma_base_config.attn_window = 72
ma_base_config.frame_chunk_size = 4

ma_base_config.env_type = 'none'

ma_base_config.height = 256
ma_base_config.width = 256
ma_base_config.action_dim = 30
ma_base_config.action_per_frame = 8
ma_base_config.obs_cam_keys = [
    'observation.images.faceImg', 'observation.images.leftImg',
    'observation.images.rightImg'
]
ma_base_config.guidance_scale = 5
ma_base_config.action_guidance_scale = 1

ma_base_config.num_inference_steps = 25
ma_base_config.video_exec_step = -1
ma_base_config.action_num_inference_steps = 50

ma_base_config.snr_shift = 5.0
ma_base_config.action_snr_shift = 1.0
# COMMENT: slice to only the arm joint positions
ma_base_config.action_slicing_ids = list(range(14, 20)) + list(range(35, 41)) + [20, 41]


# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
ma_base_config.used_action_channel_ids = list(range(14, 20)) + list(range(21, 27)) + [28, 29]
inverse_used_action_channel_ids = [
    len(ma_base_config.used_action_channel_ids)
] * ma_base_config.action_dim
for i, j in enumerate(ma_base_config.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
ma_base_config.inverse_used_action_channel_ids = inverse_used_action_channel_ids



ma_preliminary_config = EasyDict(__name__='Config: ManipArena preliminary config')
ma_preliminary_config.update(ma_base_config)
ma_preliminary_config.action_norm_method = 'quantiles'
ma_preliminary_config.norm_stat = {                                                                                                  
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.80539674, 0.0021095276, -2.1857049, -0.18749249, -0.59161705, -0.4725824, 0, -0.14783478, 0.0013418198, -2.1518216, -0.25234604, -0.15680277, -0.71049786, 0, 0.052834511, 0.049019814],                                                                                            
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16600104, 2.421458, -0.014763832, 1.1803064, 0.23785019, 0.54341221, 0, 0.81855869, 2.414953, -0.010162354, 1.2099029, 0.68181098, 0.46864223, 0, 4.18537, 4.0218706],
}

ma_preliminary_config.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/preliminary'
ma_preliminary_config.empty_emb_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/preliminary/classify_items_as_shape/empty_emb.pt"
ma_preliminary_config.wan22_finetuned_model_name_or_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_preliminary/bs16lr2.5e-5*1e-6_resume1000/checkpoints/checkpoint_step_11000'
ma_preliminary_config.enable_wandb = False 
ma_preliminary_config.load_worker = 2
ma_preliminary_config.save_interval = 500
ma_preliminary_config.gc_interval = 50
ma_preliminary_config.cfg_prob = 0.1

# Training parameters
ma_preliminary_config.learning_rate = 2.5e-5
ma_preliminary_config.beta1 = 0.9
ma_preliminary_config.beta2 = 0.95
ma_preliminary_config.weight_decay = 1e-1
ma_preliminary_config.warmup_steps = 10  # 100
ma_preliminary_config.batch_size = 1
ma_preliminary_config.min_lr = 1e-6
ma_preliminary_config.gradient_accumulation_steps = 2  # effective batch size = 2*8=16
ma_preliminary_config.num_steps = 15000
ma_preliminary_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"
ma_preliminary_config.save_root = "./checkpoints/ma_preliminary/bs16lr2.5e-5*1e-6_resume"
ma_preliminary_config.max_latent_frames = 126
# COMMENT: inference speedup
ma_preliminary_config.num_inference_steps = 20
ma_preliminary_config.video_exec_step = 10
ma_preliminary_config.action_num_inference_steps = 10
ma_preliminary_config.frame_chunk_size = 3


ma_final_config = EasyDict(__name__='Config: ManipArena preliminary config')
ma_final_config.update(ma_preliminary_config)
# COMMENT: adding the mobile dimensions
ma_final_config.action_pad_to_dim = 62
ma_final_config.action_slicing_ids = list(range(14, 20)) + list(range(35, 41)) + [20, 41] + list(range(56, 62))
ma_final_config.used_action_channel_ids = list(range(14, 20)) + list(range(21, 27)) + [28, 29] + list(range(0, 6))
ma_final_config.inverse_used_action_channel_ids = inverse_ids(ma_final_config)

# ma_final_config.norm_stat = {
#     "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.77522373, 0.0059127808, -1.7891712, -1.2205315, -0.45986843, -0.65594673, 0, -0.16240788, 0.0070571895, -1.8275874, -1.1465244, -1.2308311, -0.92357922, 0, -0.049592018, -0.019836426],
#     "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16460705, 2.374486, 1.8530178, 1.0725183, 1.2815676, 0.80510426, 0, 0.82260519, 2.5730391, 2.0616846, 1.0603113, 0.66853619, 0.68360293, 0, 4.4029055, 4.4041348],
# }

# adding the mobile dims
ma_final_config.norm_stat = {
    "q01": [-0.0015343553, -0.99886537, -4.6491623e-06, -0.10027018, -0.099322662, -0.13975513, 0, 0, 0, 0, 0, 0, 0, 0, -0.77522373, 0.0059127808, -1.7891712, -1.2205315, -0.45986843, -0.65594673, 0, -0.16240788, 0.0070571895, -1.8275874, -1.1465244, -1.2308311, -0.92357922, 0, -0.049592018, -0.019836426],
    "q99": [0, 0, 0.4700025, 0.10112506, 0.10034998, 0.1410358, 0, 0, 0, 0, 0, 0, 0, 0, 0.16460705, 2.374486, 1.8530178, 1.0725183, 1.2815676, 0.80510426, 0, 0.82260519, 2.5730391, 2.0616846, 1.0603113, 0.66853619, 0.68360293, 0, 4.4029055, 4.4041348],
}



ma_final_config.empty_emb_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/final/arrange_cup_inverted_triangle/empty_emb.pt"
ma_final_config.dataset_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/final"
# ma_final_config.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_preliminary/bs16lr2.5e-5*1e-6_resume1000/checkpoints/checkpoint_step_11000"
ma_final_config.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_final/bs16lr2.5e-5*1e-6/checkpoints/checkpoint_step_9500"
ma_final_config.save_root = "./checkpoints/ma_final/bs16lr2.5e-5*1e-6_resume12000_addMobileDims"
ma_final_config.num_steps = 10000



ma_sim_config = EasyDict(__name__='Config: ManipArena preliminary config')
ma_sim_config.update(ma_preliminary_config)
ma_sim_config.action_slicing_ids=list(range(14, 20)) + list(range(21, 27)) + [20, 27]
ma_sim_config.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_preliminary/bs16lr2.5e-5*1e-6_resume/checkpoints/checkpoint_step_5000"
ma_sim_config.save_root = "./checkpoints/ma_sim/bs8lr2.5e-5*1e-6"
ma_sim_config.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7547493, -0.00019073486, -2.4580374, -0.37174797, -0.36640739, -0.56820774, 0, -0.31948566, -0.0034928708, -2.3462658, -0.36564445, -0.73823166, -0.69638348, 0, 8.0925414e-12, 1.041886e-10],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.31033039, 2.5254452, 0.00061351777, 1.3857098, 0.462044, 0.7074461, 0, 0.99370813, 2.536622, 0.0004567389, 1.1175327, 0.33894062, 0.89322472, 0, 4.5, 4.5],
}
ma_sim_config.num_steps=5000
ma_sim_config.warmup_steps=100

# resume
ma_sim_config.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_sim/bs9lr2.5e-5*1e-6_resume/checkpoints/checkpoint_step_500"
ma_sim_config.save_root = "./checkpoints/ma_sim/bs8lr2.5e-5*1e-6_resume_2"
ma_sim_config.warmup_steps = 0
ma_sim_config.gradient_accumulation_steps = 1


