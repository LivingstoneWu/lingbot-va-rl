# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'

va_robotwin_train_cfg.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robotwin/sampled_10'
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_cfg.enable_wandb = True
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.save_interval = 1000
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1 
va_robotwin_train_cfg.gradient_accumulation_steps = 1
va_robotwin_train_cfg.num_steps = 50000 


# COMMENT: my configs

va_robotwin_train10_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train10_cfg.update(va_robotwin_train_cfg)
va_robotwin_train10_cfg.enable_wandb = False 
va_robotwin_train10_cfg.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robotwin/sampled_10'
va_robotwin_train10_cfg.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"
# va_robotwin_train10_cfg.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/robotwin_10/bs8_lr1e-5_3e-6/checkpoints/checkpoint_step_12000"
va_robotwin_train10_cfg.empty_emb_path = os.path.join(va_robotwin_train10_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train10_cfg.load_worker = 2
va_robotwin_train10_cfg.save_interval = 500
va_robotwin_train10_cfg.gc_interval = 50
va_robotwin_train10_cfg.cfg_prob = 0.1
# Training parameters
va_robotwin_train10_cfg.learning_rate = 1e-5
va_robotwin_train10_cfg.beta1 = 0.9
va_robotwin_train10_cfg.beta2 = 0.95
va_robotwin_train10_cfg.min_lr = 3e-6
va_robotwin_train10_cfg.weight_decay = 0.1
va_robotwin_train10_cfg.warmup_steps = 10
va_robotwin_train10_cfg.batch_size = 1 
va_robotwin_train10_cfg.gradient_accumulation_steps = 4
va_robotwin_train10_cfg.num_steps = 12000
va_robotwin_train10_cfg.save_root = './checkpoints/robotwin_10/bs16_lr1e-5_3e-6_resume12000'
va_robotwin_train10_cfg.grad_log_freq = 100



va_robotwin_jepatrain10_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_jepatrain10_cfg.update(va_robotwin_train10_cfg)
va_robotwin_jepatrain10_cfg.dataset_path = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robotwin/sampled_10'
# va_robotwin_jepatrain10_cfg.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/robotwin_10/bs8_lr1e-5_3e-6/checkpoints/checkpoint_step_12000"
va_robotwin_jepatrain10_cfg.wan22_pretrained_model_name_or_path = "/liujinxin/weights/lingbot-va-base"
va_robotwin_jepatrain10_cfg.empty_emb_path = os.path.join(va_robotwin_jepatrain10_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_jepatrain10_cfg.load_worker = 2
va_robotwin_jepatrain10_cfg.save_interval = 500
va_robotwin_jepatrain10_cfg.gc_interval = 50
va_robotwin_jepatrain10_cfg.cfg_prob = 0.1
# Training parameters
va_robotwin_jepatrain10_cfg.learning_rate = 1e-5
va_robotwin_jepatrain10_cfg.beta1 = 0.9
va_robotwin_jepatrain10_cfg.beta2 = 0.95
va_robotwin_jepatrain10_cfg.min_lr = 3e-6
va_robotwin_jepatrain10_cfg.weight_decay = 0.1
va_robotwin_jepatrain10_cfg.warmup_steps = 10
va_robotwin_jepatrain10_cfg.batch_size = 1 
va_robotwin_jepatrain10_cfg.gradient_accumulation_steps = 2
va_robotwin_jepatrain10_cfg.num_steps = 20000
va_robotwin_jepatrain10_cfg.save_root = './checkpoints/robotwin_10_jepa/bs16_lr1e-5_3e-6'
va_robotwin_jepatrain10_cfg.grad_log_freq = 100
# JEPA params
va_robotwin_jepatrain10_cfg.jepa_loss_enabled = True
va_robotwin_jepatrain10_cfg.jepa_loss_layer    = 20
va_robotwin_jepatrain10_cfg.jepa_head_type     = "linear"
va_robotwin_jepatrain10_cfg.jepa_loss_weight   = 0.1
va_robotwin_jepatrain10_cfg.jepa_loss_t_max    = 1.0

va_robotwin_jepatrain_full_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_jepatrain_full_cfg.update(va_robotwin_jepatrain10_cfg)
va_robotwin_jepatrain_full_cfg.dataset_path = "/luhongchao/shared/dataset/robotwin_converted/full"
va_robotwin_jepatrain_full_cfg.wan22_pretrained_model_name_or_path = "/luhongchao/shared/weights/lingbot-va-base"
# va_robotwin_jepatrain_full_cfg.wan22_pretrained_model_name_or_path = "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/robotwin_full_jepa/bs32_lr1e-5_3e-6/checkpoints/checkpoint_step_28000"
va_robotwin_jepatrain_full_cfg.learning_rate = 1e-5
va_robotwin_jepatrain_full_cfg.min_lr = 1e-5
va_robotwin_jepatrain_full_cfg.empty_emb_path = "/luhongchao/shared/dataset/robotwin_converted/empty_emb.pt"
va_robotwin_jepatrain_full_cfg.save_interval = 5000
va_robotwin_jepatrain_full_cfg.max_latent_frames = 68
va_robotwin_jepatrain_full_cfg.batch_size = 1
va_robotwin_jepatrain_full_cfg.gradient_accumulation_steps = 4
va_robotwin_jepatrain_full_cfg.num_steps = 50000
# va_robotwin_jepatrain_full_cfg.num_steps = 22000
va_robotwin_jepatrain_full_cfg.save_root = './checkpoints/robotwin_full_jepa/bs32_lr1e-5'




# COMMENT: RL configs
robotwin_qgf_v1_cfg_place_can_basket_generated = EasyDict(__name__='Config: VA robotwin train')
robotwin_qgf_v1_cfg_place_can_basket_generated.update(va_robotwin_jepatrain10_cfg)
robotwin_qgf_v1_cfg_place_can_basket_generated.dataset_path = "/luhongchao/shared/dataset/robotwin_converted/place_can_basket_robotwin_generated_success_100"
robotwin_qgf_v1_cfg_place_can_basket_generated.empty_emb_path = "/luhongchao/shared/dataset/robotwin_converted/empty_emb.pt"
robotwin_qgf_v1_cfg_place_can_basket_generated.frame_chunk_size = 2
robotwin_qgf_v1_cfg_place_can_basket_generated.wan22_pretrained_model_name_or_path = "/luhongchao/shared/weights/lingbot-va-posttrain-robotwin"
robotwin_qgf_v1_cfg_place_can_basket_generated.num_inference_steps = 20
robotwin_qgf_v1_cfg_place_can_basket_generated.video_exec_step = 20
robotwin_qgf_v1_cfg_place_can_basket_generated.action_num_inference_steps = 10
robotwin_qgf_v1_cfg_place_can_basket_generated.learning_rate = 1e-5
robotwin_qgf_v1_cfg_place_can_basket_generated.min_lr = 1e-5
robotwin_qgf_v1_cfg_place_can_basket_generated.save_interval = 1000
robotwin_qgf_v1_cfg_place_can_basket_generated.max_latent_frames = 68
robotwin_qgf_v1_cfg_place_can_basket_generated.batch_size = 2
robotwin_qgf_v1_cfg_place_can_basket_generated.jepa_loss_enabled = False
robotwin_qgf_v1_cfg_place_can_basket_generated.gradient_accumulation_steps = 1
robotwin_qgf_v1_cfg_place_can_basket_generated.num_steps = 10000
robotwin_qgf_v1_cfg_place_can_basket_generated.save_root = './checkpoints/robotwin_generated_100_validation/bs16_lr1e-5'

robotwin_place_can_basket_official_dataset = EasyDict(__name__='Config: VA robotwin train')
robotwin_place_can_basket_official_dataset.update(robotwin_qgf_v1_cfg_place_can_basket_generated)
robotwin_place_can_basket_official_dataset.dataset_path = "/luhongchao/shared/dataset/robotwin_converted/lingbot_official_place_can_basket"
robotwin_place_can_basket_official_dataset.wan22_pretrained_model_name_or_path = "/luhongchao/shared/weights/lingbot-va-posttrain-robotwin"
robotwin_place_can_basket_official_dataset.batch_size = 1
robotwin_place_can_basket_official_dataset.gradient_accumulation_steps = 2
robotwin_place_can_basket_official_dataset.num_steps = 10000
robotwin_place_can_basket_official_dataset.min_lr = 3e-6
robotwin_place_can_basket_official_dataset.save_root = './checkpoints/posttrain_official_dataset_place_can_basket/bs16_lr1e-5'
robotwin_place_can_basket_official_dataset.norm_stat = {
    "q01": [-0.30003312, -0.3138428, 0.87322855, 0.65376788, -0.27025825, -0.047022339, -0.10308055, 0.01576671, -0.3860966, 0.87763625, 0.0043548741, -0.65071458, -0.44845328, -0.74115753, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "q99": [-0.023393035, 0.091135211, 1.112532, 0.99992442, 0.10347752, 0.70785922, 0.72107124, 0.33953321, 0.084412366, 1.224782, 0.7524699, 0.70665616, 0.32905793, 0.99893397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
}

robotwin_qgf_v1_cfg_place_can_basket_200rollout_50successGenerated = EasyDict(__name__='Config: VA robotwin train')
robotwin_qgf_v1_cfg_place_can_basket_200rollout_50successGenerated.update(robotwin_qgf_v1_cfg_place_can_basket_generated)
robotwin_qgf_v1_cfg_place_can_basket_200rollout_50successGenerated.dataset_path = "/luhongchao/shared/dataset/robotwin_converted/place_can_basket_200rollout_50successGenerated"
robotwin_qgf_v1_cfg_place_can_basket_200rollout_50successGenerated.save_root = './checkpoints/robotwin_place_can_basket_200rollout_50successGenerated/bs6_lr1e-5'
robotwin_qgf_v1_cfg_place_can_basket_200rollout_50successGenerated.jepa_loss_enabled = False