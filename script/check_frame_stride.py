import torch
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = torch.load(path, map_location='cpu')

print(f"File: {path.name}")
print(f"Keys: {list(data.keys())}")

for key, val in data.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
    elif isinstance(val, list):
        print(f"  {key}: list of {len(val)} items → {val[:5]}{'...' if len(val)>5 else ''}")
    else:
        print(f"  {key}: {val}")

if 'frame_ids' in data:
    ids = data['frame_ids']
    print(f"\nframe_ids summary:")
    print(f"  first={ids[0]}, last={ids[-1]}, count={len(ids)}")
    if len(ids) >= 2:
        stride = ids[1] - ids[0]
        print(f"  stride (ids[1]-ids[0]) = {stride}")
        print(f"  → action_per_frame should be {stride * 4}")
