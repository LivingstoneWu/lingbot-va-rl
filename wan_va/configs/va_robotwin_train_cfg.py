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
robotwin_qgf_v1_cfg_place_can_basket_generated.dataset_path = "/luhongchao/shared/dataset/robotwin_rl_converted/place_can_basket_robotwin_generated_100"
robotwin_qgf_v1_cfg_place_can_basket_generated.empty_emb_path = "/luhongchao/shared/dataset/robotwin_converted/empty_emb.pt"
robotwin_qgf_v1_cfg_place_can_basket_generated.frame_chunk_size = 2
robotwin_qgf_v1_cfg_place_can_basket_generated.wan22_pretrained_model_name_or_path = "/luhongchao/shared/weights/lingbot-va-posttrain-robotwin"
