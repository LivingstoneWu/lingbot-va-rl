import torch
from wan_va.modules.utils import load_vae, WanVAEStreamingWrapper

# ---- setup ----
device = "cuda:0"
vae_path = "/liujinxin/weights/lingbot-va-base/vae/"   # change this
dtype = torch.float32

vae = load_vae(vae_path, torch_dtype=dtype, torch_device=device)
vae_wrapper = WanVAEStreamingWrapper(vae)

# synthetic video: B=1, C=3, T frames
B, C, T, H, W = 1, 3, 13, 256, 320
video = torch.randn(B, C, T, H, W, device=device, dtype=dtype)

# ---- 1) full encode once ----
with torch.no_grad():
    full_enc = vae.encode(video).latent_dist.mode()   # [B, 2*z_dim, F_lat, H_lat, W_lat]
print("full_enc.shape:", tuple(full_enc.shape))
print("full latent frames:", full_enc.shape[2])

# ---- 2) online encode one frame at a time ----
vae_wrapper.clear_cache()
outs = []
frame_t = video[:, :, :1]             # one frame
out_t = vae_wrapper.encode_chunk(frame_t)
outs.append(out_t)
print(f"t=1, out_t.shape={tuple(out_t.shape)}, new_lat_frames={out_t.shape[2]}")
for t in range(1, (T-1)//4 + 1):
    frame_t = video[:, :, t:t+4]             # one frame
    out_t = vae_wrapper.encode_chunk(frame_t)
    outs.append(out_t)
    print(f"t={t:02d}, out_t.shape={tuple(out_t.shape)}, new_lat_frames={out_t.shape[2]}")

stream_enc = torch.cat(outs, dim=2) if len(outs) > 0 else None
print("stream_enc.shape:", tuple(stream_enc.shape))
print("stream total latent frames:", stream_enc.shape[2])

# ---- 3) compare full vs streaming totals ----
print("F_lat match?", stream_enc.shape[2] == full_enc.shape[2])

# optional: numerical check if same shape
if stream_enc.shape == full_enc.shape:
    max_abs = (stream_enc - full_enc).abs().max().item()
    print("max |stream - full|:", max_abs)
