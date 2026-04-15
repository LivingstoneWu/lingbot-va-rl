import torch, glob, os
dataset_root = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/ur5e/stack_color_blocks_action_corrected"
pths = glob.glob(os.path.join(dataset_root, "latents", "chunk-*", "observation.images.*", "episode_*.pth"))
assert pths, "No latent pth found"
sample = torch.load(pths[0], weights_only=False)
emb = sample["text_emb"]
assert emb is not None, "Sample text_emb is None; ensure metadata tasks/text exists during extraction"
empty = torch.zeros_like(emb,dtype=torch.bfloat16)
torch.save(empty, os.path.join(dataset_root, "empty_emb.pt"))
print("saved:", os.path.join(dataset_root, "empty_emb.pt"), "shape:", tuple(empty.shape), "dtype:", empty.dtype)