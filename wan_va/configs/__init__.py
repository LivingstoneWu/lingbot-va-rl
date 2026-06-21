# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import *
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .rc_aloha_configs import *
from .rc_ur5_configs import *
from .ma_configs import *
from .rc_arx5_configs import *
from .rc_franka_configs import *

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'rc_aloha_pencil_case': rc_aloha_pencil_case,
    'rc_single_set_the_plates_config_train': rc_single_set_the_plates_config_train,
    'rc_ur5_base_config': rc_ur5_base_config,
    'rc_single_stack_color_blocks_config_train': rc_single_stack_color_blocks_config_train,
    'rc_single_arrange_fruits_config_train': rc_single_arrange_fruits_config_train,
    'local_ur5_stack_color_blocks_config': local_ur5_stack_color_blocks_config, 
    'ma_preliminary_config': ma_preliminary_config,
    'rc_arx5_base_config': rc_arx5_base_config,
    'ma_sim_config': ma_sim_config,
    'rc_ur5_shred_papers': rc_ur5_shred_papers,
    'rc_arx5_arrange_flowers': rc_arx5_arrange_flowers,
    'rc_arx5_place_shoes': rc_arx5_place_shoes,
    'rc_aloha_plug_in_network_cable': rc_aloha_plug_in_network_cable,
    'ma_final_config': ma_final_config,
    'rc_arx5_wipe_table': rc_arx5_wipe_table,
    'rc_franka_press_button': rc_franka_press_button,
    'va_robotwin_train10_cfg': va_robotwin_train10_cfg,
    'va_robotwin_jepatrain10_cfg': va_robotwin_jepatrain10_cfg,
    'rc_aloha_base_config': rc_aloha_base_config,
    'va_robotwin_jepatrain_full_cfg': va_robotwin_jepatrain_full_cfg,
    'robotwin_qgf_v1_cfg_place_can_basket_generated': robotwin_qgf_v1_cfg_place_can_basket_generated,
}