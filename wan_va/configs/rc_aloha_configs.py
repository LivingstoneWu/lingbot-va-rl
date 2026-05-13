# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
import os

from .shared_config import va_shared_cfg

rc_aloha_base_config = EasyDict(__name__='Config: RC ALOHA base')
rc_aloha_base_config.update(va_shared_cfg)

rc_aloha_base_config.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"

# COMMENT: the latent chunk size, extraction 501 chunk_size -> (501-1)/4 + 1 = 126 frames
rc_aloha_base_config.max_latent_frames = 126


# COMMENT: These are for inference
rc_aloha_base_config.attn_window = 72
rc_aloha_base_config.frame_chunk_size = 4

rc_aloha_base_config.env_type = 'none'

rc_aloha_base_config.height = 256
rc_aloha_base_config.width = 256
rc_aloha_base_config.action_dim = 30
rc_aloha_base_config.action_per_frame = 12
rc_aloha_base_config.obs_cam_keys = [
    'observation.images.top', 'observation.images.left_wrist',
    'observation.images.right_wrist'
]
rc_aloha_base_config.guidance_scale = 5
rc_aloha_base_config.action_guidance_scale = 1

rc_aloha_base_config.num_inference_steps = 10
rc_aloha_base_config.video_exec_step = -1
rc_aloha_base_config.action_num_inference_steps = 20

rc_aloha_base_config.snr_shift = 5.0
rc_aloha_base_config.action_snr_shift = 1.0


# COMMENT: inverse_used_action_channel_ids maps the 30 action dimensions to the indices of the used action channels. 
# COMMENT: not used ones are mapped to len(used_ids), or action_dim + 1, facilitating later padding.
rc_aloha_base_config.used_action_channel_ids = list(range(14, 20)) + list(range(21,27)) + list(range(28, 30))
inverse_used_action_channel_ids = [
    len(rc_aloha_base_config.used_action_channel_ids)
] * rc_aloha_base_config.action_dim
for i, j in enumerate(rc_aloha_base_config.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
rc_aloha_base_config.inverse_used_action_channel_ids = inverse_used_action_channel_ids



rc_aloha_pencil_case = EasyDict(__name__='Config: RC ALOHA pencil case')
rc_aloha_pencil_case.update(rc_aloha_base_config)
rc_aloha_pencil_case.action_norm_method = 'quantiles'
rc_aloha_pencil_case.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.43822816, -0.000627984, -1.3246624, -0.90175295, -0.1313882, -0.19809407, 0, -0.072915919, -0.00045354399, -1.5296671, -0.085161611, -0.15870552, -1.942686, 0, -0.00079999998, -0.0034],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17438766, 1.8653436, 0.0066287201, 0.15326571, 1.194129, 1.8375188, 0, 0.73050237, 2.1186435, 0.0090185478, 1.7354861, 1.1026527, 0.30600265, 0, 0.059700001, 0.067100003],
}

rc_aloha_pencil_case.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case_trim'
rc_aloha_pencil_case.empty_emb_path = os.path.join(rc_aloha_pencil_case.dataset_path, 'empty_emb.pt')
rc_aloha_pencil_case.enable_wandb = False 
rc_aloha_pencil_case.load_worker = 2
rc_aloha_pencil_case.save_interval = 500
rc_aloha_pencil_case.gc_interval = 50
rc_aloha_pencil_case.cfg_prob = 0.1

# Training parameters
rc_aloha_pencil_case.learning_rate = 7e-6
rc_aloha_pencil_case.beta1 = 0.9
rc_aloha_pencil_case.beta2 = 0.95
rc_aloha_pencil_case.weight_decay = 1e-1
rc_aloha_pencil_case.warmup_steps = 50
rc_aloha_pencil_case.batch_size = 1
rc_aloha_pencil_case.min_lr = 1e-6
rc_aloha_pencil_case.gradient_accumulation_steps = 4  # effective batch size = 2*8=16
rc_aloha_pencil_case.num_steps = 6000
rc_aloha_pencil_case.save_root = "./checkpoints/rc_aloha_pencil_case/bs16lr7e-6_resume8000"
rc_aloha_pencil_case.infer_mode = 'server'
rc_aloha_pencil_case.frame_chunk_size = 4
rc_aloha_pencil_case.action_per_frame = 12
rc_aloha_pencil_case.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/rc_aloha_pencil_case/bs16lr2.5e-5/checkpoints/checkpoint_step_2000"

rc_aloha_plug_in_network_cable = EasyDict(__name__='Config: RC ALOHA pencil case')
rc_aloha_plug_in_network_cable.update(rc_aloha_pencil_case)
rc_aloha_plug_in_network_cable.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"
rc_aloha_plug_in_network_cable.save_root = "./checkpoints/rc_aloha_network_cable/bs8lr2.5e-5"
rc_aloha_plug_in_network_cable.empty_emb_path = os.path.join(rc_aloha_plug_in_network_cable.dataset_path, 'empty_emb.pt')
rc_aloha_plug_in_network_cable.dataset_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/plug_in_network_cable_trim"
rc_aloha_plug_in_network_cable.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.62589073, 0.013065556, -1.4073645, -0.94768018, -0.1030766, -0.43350086, 0, -0.050535269, 0.041202728, -1.6591512, 0, -1.1627299, -1.3525229, 0, -0.0016, -0.0011],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.24993764, 1.9491228, 0.0068380479, 0.23619176, 1.2158643, 1.4879906, 0, 1.3318669, 2.2066486, -0.023880836, 1.6672375, 1.2199985, 0.24698959, 0, 0.048099998, 0.045699999],
}