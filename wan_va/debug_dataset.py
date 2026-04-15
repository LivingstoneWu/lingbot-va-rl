# run from wan_va/ directory
import sys
sys.path.insert(0, '.')
from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import construct_lerobot_multi_processor
from pathlib import Path
import os

config = VA_CONFIGS['rc_ur5_set_the_plates']  # whichever config you're using

datasets = construct_lerobot_multi_processor(config, num_init_worker=1)
dset = datasets[0]  # inspect first dataset

print(f"Total episodes in meta : {len(dset.meta.episodes)}")
print(f"new_metas (valid items): {len(dset.new_metas)}")
print()

# Check the first 3 episodes in detail
for i, (key, value) in enumerate(list(dset.meta.episodes.items())[:3]):
    ep_idx = value["episode_index"]
    action_config = value.get("action_config", "MISSING")
    print(f"Episode {ep_idx}: action_config = {action_config}")

    if action_config == "MISSING" or len(action_config) == 0:
        print("  → filtered out: action_config missing or empty")
        continue

    for acfg in action_config:
        start, end = acfg["start_frame"], acfg["end_frame"]
        episode_chunk = dset.meta.get_episode_chunk(ep_idx)
        latent_base = Path(dset.latent_path) / f"chunk-{episode_chunk:03d}"
        for cam in dset.used_video_keys:
            f = latent_base / cam / f"episode_{ep_idx:06d}_{start}_{end}.pth"
            print(f"  [{cam}] {f}")
            print(f"  → exists: {f.exists()}")
