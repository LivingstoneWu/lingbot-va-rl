import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_q01_q99_parquet(folder_pattern, key="action", sample_ratio=1.0):
    files = sorted(glob.glob(folder_pattern))
    
    all_data = []

    for f in tqdm(files):
        df = pd.read_parquet(f)

        if key not in df:
            continue

        data = df[key].values  # 可能是 list / ndarray

        # 转成 numpy
        data = np.stack(data)  # shape: [T, D] or [T, ...]

        # flatten 时间维
        data = data.reshape(-1, data.shape[-1])

        # 可选：采样（防止爆内存）
        if sample_ratio < 1.0:
            n = int(len(data) * sample_ratio)
            idx = np.random.choice(len(data), n, replace=False)
            data = data[idx]

        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)

    q01 = np.quantile(all_data, 0.01, axis=0)
    q99 = np.quantile(all_data, 0.99, axis=0)

    return q01, q99


folder = "/liujinxin/code/lhc/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case/data/chunk-000/episode_*.parquet"

q01, q99 = compute_q01_q99_parquet(folder)

print("q01:", q01)
print("q99:", q99)
