import json
import torch
import pyarrow.parquet as pq
from pathlib import Path

DATASET = Path("/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/set_the_plates")
LATENTS = DATASET / "latents"

# Get per-episode row counts from parquet
ep_lengths = {}
for pq_file in sorted((DATASET / "data").rglob("*.parquet")):
    table = pq.read_table(pq_file, columns=["episode_index"])
    agg = table.group_by("episode_index").aggregate([("episode_index", "count")]).to_pydict()
    for ep_idx, count in zip(agg["episode_index"], agg["episode_index_count"]):
        ep_lengths[ep_idx] = count

# Check each action_config entry against its actual .pth file
with open(DATASET / "meta/episodes.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        ep_idx = ep["episode_index"]
        ep_len = ep_lengths.get(ep_idx, 0)
        chunk = ep_idx // 1000  # or however your chunks are organised

        for acfg in ep.get("action_config", []):
            sf, ef = acfg["start_frame"], acfg["end_frame"]

            # find the .pth file
            pth_glob = list(LATENTS.glob(
                f"chunk-{chunk:03d}/*/episode_{ep_idx:06d}_{sf}_{ef}.pth"
            ))
            if not pth_glob:
                print(f"MISSING pth: episode={ep_idx} {sf}-{ef}")
                continue

            data = torch.load(pth_glob[0], map_location="cpu")
            frame_ids = data["frame_ids"]

            act_shift     = int(frame_ids[0] - sf)
            frame_stride  = int(frame_ids[1] - frame_ids[0])
            latent_frame_num = (len(frame_ids) - 1) // 4 + 1
            required      = latent_frame_num * frame_stride * 4
            available     = (ef - sf - act_shift) + frame_stride * 4

            if available < required:
                print(
                    f"BAD: episode={ep_idx} sf={sf} ef={ef} ep_len={ep_len} "
                    f"frame_ids[0:2]={list(frame_ids[:2])} "
                    f"stride={frame_stride} latent_frames={latent_frame_num} "
                    f"required={required} available={available} "
                    f"shortfall={required - available}"
                )
