import torch, glob, os
dataset_root = "/luhongchao/shared/dataset/robotwin_converted/lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50"
pths = glob.glob(os.path.join(dataset_root, "latents", "chunk-*", "observation.images.*", "episode_*.pth"))
assert pths, "No latent pth found"
sample = torch.load(pths[0], weights_only=False)
emb = sample["text_emb"]
assert emb is not None, "Sample text_emb is None; ensure metadata tasks/text exists during extraction"
empty = torch.zeros_like(emb,dtype=torch.bfloat16)
torch.save(empty, os.path.join(dataset_root, "empty_emb.pt"))
print("saved:", os.path.join(dataset_root, "empty_emb.pt"), "shape:", tuple(empty.shape), "dtype:", empty.dtype)