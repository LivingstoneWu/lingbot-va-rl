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
rc_aloha_base_config.width = 256
rc_aloha_base_config.action_dim = 30
rc_aloha_base_config.action_per_frame = 8
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
rc_aloha_base_config.used_action_channel_ids = list(range(0, 20)) + list(range(21,27)) + list(range(28, 30))
inverse_used_action_channel_ids = [
    len(rc_aloha_base_config.used_action_channel_ids)
] * rc_aloha_base_config.action_dim
for i, j in enumerate(rc_aloha_base_config.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
rc_aloha_base_config.inverse_used_action_channel_ids = inverse_used_action_channel_ids



rc_aloha_pencil_case_config_train = EasyDict(__name__='Config: RC ALOHA pencil case')
rc_aloha_pencil_case_config_train.update(rc_aloha_base_config)
rc_aloha_pencil_case_config_train.action_norm_method = 'quantiles'
rc_aloha_pencil_case_config_train.norm_stat = {
    "q01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.43822816, -0.000627984, -1.3246624, -0.90175295, -0.1313882, -0.19809407, 0, -0.072915919, -0.00045354399, -1.5296671, -0.085161611, -0.15870552, -1.942686, 0, -0.00079999998, -0.0034],
    "q99": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17438766, 1.8653436, 0.0066287201, 0.15326571, 1.194129, 1.8375188, 0, 0.73050237, 2.1186435, 0.0090185478, 1.7354861, 1.1026527, 0.30600265, 0, 0.059700001, 0.067100003],
}

rc_aloha_pencil_case_config_train.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case'
rc_aloha_pencil_case_config_train.empty_emb_path = os.path.join(rc_aloha_pencil_case_config_train.dataset_path, 'empty_emb.pt')
rc_aloha_pencil_case_config_train.enable_wandb = False 
rc_aloha_pencil_case_config_train.load_worker = 2
rc_aloha_pencil_case_config_train.save_interval = 500
rc_aloha_pencil_case_config_train.gc_interval = 50
rc_aloha_pencil_case_config_train.cfg_prob = 0.1

# Training parameters
rc_aloha_pencil_case_config_train.learning_rate = 2.5e-5
rc_aloha_pencil_case_config_train.beta1 = 0.9
rc_aloha_pencil_case_config_train.beta2 = 0.95
rc_aloha_pencil_case_config_train.weight_decay = 1e-1
rc_aloha_pencil_case_config_train.warmup_steps = 50
rc_aloha_pencil_case_config_train.batch_size = 1
rc_aloha_pencil_case_config_train.min_lr = 1e-6
rc_aloha_pencil_case_config_train.gradient_accumulation_steps = 2  # effective batch size = 2*8=16
rc_aloha_pencil_case_config_train.num_steps = 10000 
rc_aloha_pencil_case_config_train.save_root = "./checkpoints/rc_aloha_pencil_case/bs16lr2.5e-5"

rc_aloha_pencil_case_config_inf = EasyDict(__name__='Config: pencil inference')
rc_aloha_pencil_case_config_inf.update(rc_aloha_pencil_case_config_train)
rc_aloha_pencil_case_config_inf.save_root = './inf_out'
rc_aloha_pencil_case_config_inf.infer_mode = 'server'

rc_aloha_qr_code_config_inf = EasyDict(__name__='Config: QR code inference')
rc_aloha_qr_code_config_inf.update(rc_aloha_pencil_case_config_inf)
rc_aloha_qr_code_config_inf.norm_stat={
  "q01": [ 3.33133787e-02, -1.74428686e-01,  1.27174839e-01, -1.47408545e-01,
  5.67829728e-01, -4.01000917e-01,  2.99636181e-02,  5.18027060e-02,
 -3.86058935e-03,  1.14107616e-01, -6.70769691e-01, -6.50196075e-01,
 -1.26051053e-01, -2.17450768e-01, -4.88634884e-01,  0.00000000e+00,
 -1.41753435e+00, -1.14087248e+00, -4.99264717e-01, -1.61858749e+00, 0.0,
 -6.22053035e-02,  2.61659990e-03, -1.58443856e+00, -4.12969261e-01,
 -3.23010534e-01, -1.59202671e+00,  0.0, 9.99999975e-05,  1.09999999e-03],
  
"q99": [0.3577207,  0.01039131, 0.42667967, 0.68634874, 
        0.98858124, 0.09494539, 0.8047871,  0.39639464, 
        0.2131722,  0.4349674,  0.6861365,  0.9864144, 
        0.22613631, 0.7798252,  0.14058119, 1.9815686,  
        0.00570419, 1.7474004,  1.2157596,  1.7655421, 0.0, 
        0.65437675, 2.0980945,  0.00711715, 0.74620634,
        1.13618,    0.4334725, 0.0, 0.069124,   0.1018315, ]
}
rc_aloha_qr_code_config_inf.frame_chunk_size = 4
rc_aloha_qr_code_config_inf.action_per_frame = 8
