from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.dataset import MultiLatentLeRobotDataset
from torch.utils.data import DataLoader
from wan_va.configs import VA_CONFIGS
import time

config = VA_CONFIGS['rc_ur5_stack_blocks_train']
dataset = MultiLatentLeRobotDataset(config=config)
loader = DataLoader(dataset, batch_size=1, num_workers=config.load_worker)

for i, batch in enumerate(loader):
    t = time.time()
    _ = next(iter(loader))  # trigger actual I/O
    print(f"batch {i}: {time.time() - t:.2f}s")
    if i >= 10:
        break