"""Generate images with BagelForTraining using Classifier-Free Guidance (CFG).

Matches the original vllm-omni inference pipeline EXACTLY:
  * 3-branch CFG: cfg_text_scale=4.0, cfg_img_scale=1.5
    For text2img (no image input), cfg_img branch == gen branch,
    so effective formula is:  v = 5.5*v_cond - 4.5*v_uncond
  * cfg_interval = [0.4, 1.0], cfg_renorm_type = "global", cfg_renorm_min = 0.0
  * Timestep: linspace(1, 0, num_steps) → num_steps-1 actual denoising steps
  * timestep_shift = 3.0
  * Default seed = 52  (bagel_single_stage.yaml)

Dependencies: torch, PIL, transformers (via bagel_model)
NO dependency on vllm or vllm-omni.
"""
import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusers"))

from bagel_model import (
    BagelForTraining,
    AutoEncoder,
    load_ae,
    load_tokenizer,
    get_flattened_position_ids,
)

MODEL_PATH = "/proj-tango-pvc/users/zhipeng.wang/workspace/models/BAGEL-7B-MoT"
OUTPUT_DIR = "/proj-tango-pvc/users/zhipeng.wang/workspace/verl/examples/flowgrpo_trainer/outputs"
DEVICE = "cuda"
DTYPE = torch.bfloat16

NUM_STEPS = 50
TIMESTEP_SHIFT = 3.0
CFG_TEXT_SCALE = 4.0
CFG_IMG_SCALE = 1.5
CFG_INTERVAL = (0.4, 1.0)
CFG_RENORM_MIN = 0.0
CFG_RENORM_TYPE = "global"
IMG_H, IMG_W = None, None  # auto-detect from model.config.max_latent_size


def unpatchify(patches, h, w, ps, c):
    B = patches.shape[0]
    return torch.einsum("bhwpqc->bchpwq",
                        patches.view(B, h, w, ps, ps, c)).contiguous().view(B, c, h * ps, w * ps)


@torch.no_grad()
def generate_images_cfg(model, vae, text_token_ids, latent_pos_ids, img_h, img_w, seed=52):
    """Full denoising with 3-branch CFG + renorm (matches vllm-omni pipeline)."""
    model.eval()
    B = 1 if text_token_ids is None else text_token_ids.shape[0]
    L_latent = latent_pos_ids.shape[1]

    # Match vllm-omni: generate noise on CPU in float32, then move to CUDA.
    # CPU and CUDA RNG produce different numbers even with the same seed.
    torch.manual_seed(seed)
    x_t = torch.randn(B, L_latent, model.config.patch_latent_dim).to(DEVICE)

    # vllm-omni: linspace(1, 0, num_timesteps) with num_timesteps=50 → 49 actual steps
    timesteps = torch.linspace(1, 0, NUM_STEPS, device=DEVICE)
    timesteps = TIMESTEP_SHIFT * timesteps / (1 + (TIMESTEP_SHIFT - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]
    timesteps = timesteps[:-1]

    # vllm-omni wraps the entire denoising loop in autocast;
    # x_t stays float32 while model forward runs in bfloat16.
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        for i, t in enumerate(timesteps):
            ts = torch.full((B,), t.item(), device=DEVICE, dtype=DTYPE)

            v_cond = model(
                hidden_states=x_t, timestep=ts,
                text_token_ids=text_token_ids, latent_pos_ids=latent_pos_ids,
            )[0]

            in_cfg_window = t.item() > CFG_INTERVAL[0] and t.item() <= CFG_INTERVAL[1]
            cfg_text_scale = CFG_TEXT_SCALE if in_cfg_window else 1.0
            cfg_img_scale = CFG_IMG_SCALE if in_cfg_window else 1.0
            use_cfg = cfg_text_scale > 1.0

            if use_cfg:
                v_uncond = model(
                    hidden_states=x_t, timestep=ts,
                    text_token_ids=None, latent_pos_ids=latent_pos_ids,
                )[0]

                v_text = v_uncond + cfg_text_scale * (v_cond - v_uncond)
                cfg_img_v_t = v_cond
                v_guided = cfg_img_v_t + cfg_img_scale * (v_text - cfg_img_v_t)

                if CFG_RENORM_TYPE == "global":
                    norm_cond = torch.norm(v_cond.float())
                    norm_guided = torch.norm(v_guided.float())
                elif CFG_RENORM_TYPE == "channel":
                    norm_cond = torch.norm(v_cond.float(), dim=-1, keepdim=True)
                    norm_guided = torch.norm(v_guided.float(), dim=-1, keepdim=True)
                else:
                    raise ValueError(f"Unsupported renorm type: {CFG_RENORM_TYPE}")
                scale = (norm_cond / (norm_guided + 1e-8)).clamp(min=CFG_RENORM_MIN, max=1.0)
                v_t = v_guided * scale
            else:
                v_t = v_cond

            x_t = x_t - v_t * dts[i]

    latent = unpatchify(x_t, img_h, img_w, model.config.latent_patch_size, model.config.latent_channel)
    pixels = vae.decode(latent.float())
    pixels = ((pixels * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    return pixels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading BAGEL model...")
    model = BagelForTraining.from_pretrained(MODEL_PATH, torch_dtype=DTYPE).to(DEVICE)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    print("Loading VAE...")
    ae_path = os.path.join(MODEL_PATH, "ae.safetensors")
    vae, _ = load_ae(ae_path)
    vae = vae.to(DEVICE).eval()

    print("Loading tokenizer...")
    tokenizer, new_token_ids = load_tokenizer(MODEL_PATH)

    img_h = IMG_H if IMG_H is not None else model.config.max_latent_size
    img_w = IMG_W if IMG_W is not None else model.config.max_latent_size
    latent_ds = model.config.latent_patch_size * model.config.vae_downsample
    H_px = img_h * latent_ds
    W_px = img_w * latent_ds
    vae_pos_ids = get_flattened_position_ids(
        H_px, W_px, latent_ds, model.config.max_latent_size,
    ).to(DEVICE)

    prompts = [
        "A cute cat",
    ]

    print(f"\nGenerating {len(prompts)} images at {H_px}x{W_px}px (latent {img_h}x{img_w})")
    print(f"  CFG: text_scale={CFG_TEXT_SCALE}, img_scale={CFG_IMG_SCALE}, "
          f"interval={CFG_INTERVAL}, renorm={CFG_RENORM_TYPE}")
    print(f"  Steps: {NUM_STEPS} (linspace→{NUM_STEPS-1} actual), shift={TIMESTEP_SHIFT}\n")

    for idx, prompt in enumerate(prompts):
        text_ids = tokenizer.encode(prompt)
        full_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
        text_token_ids = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
        pos_ids = vae_pos_ids.unsqueeze(0)

        print(f"  [{idx}] '{prompt}'")
        pixels = generate_images_cfg(model, vae, text_token_ids, pos_ids, img_h, img_w, seed=52 + idx)
        img = Image.fromarray(pixels[0].permute(1, 2, 0).cpu().numpy())

        path = os.path.join(OUTPUT_DIR, f"sample{idx}_cfg.png")
        img.save(path)
        print(f"      Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
