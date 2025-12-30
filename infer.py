import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from diffusers import UNet2DModel, DDPMScheduler

# -------------------------
# CONFIG (MUST MATCH TRAINING)
# -------------------------
BASE_DIR = Path(__file__).parent
IMAGE_SIZE = 128
DIFFUSION_STEPS = 12
USE_RGBA = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# UTILITIES
# -------------------------
def img_to_tensor(img):
    arr = np.array(img).astype(np.float32)
    if USE_RGBA:
        arr = arr[..., :4]
    else:
        arr = arr[..., :3]
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_img(t):
    t = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = ((t + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    mode = "RGBA" if USE_RGBA else "RGB"
    return Image.fromarray(t, mode)


# -------------------------
# MODEL
# -------------------------
def load_model(checkpoint_path):
    in_ch = 9 if not USE_RGBA else 12
    out_ch = 3 if not USE_RGBA else 4

    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=in_ch,
        out_channels=out_ch,
        layers_per_block=2,
        block_out_channels=(64, 128, 128),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# -------------------------
# DIFFUSION INFERENCE
# -------------------------
@torch.no_grad()
def generate_mid(model, frame_a, frame_b):
    scheduler = DDPMScheduler(
        num_train_timesteps=DIFFUSION_STEPS,
        beta_schedule="squaredcos_cap_v2"
    )
    scheduler.set_timesteps(DIFFUSION_STEPS)

    sample = torch.randn_like(frame_a)

    for t in scheduler.timesteps:
        model_input = torch.cat([sample, frame_a, frame_b], dim=1)
        pred_noise = model(model_input, t).sample
        sample = scheduler.step(pred_noise, t, sample).prev_sample

    return sample


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame_a", required=True)
    parser.add_argument("--frame_b", required=True)
    parser.add_argument("--output", default="mid.png")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = BASE_DIR / checkpoint

    model = load_model(checkpoint)

    A = Image.open(args.frame_a).convert("RGB")
    B = Image.open(args.frame_b).convert("RGB")

    frame_a = img_to_tensor(A).to(DEVICE)
    frame_b = img_to_tensor(B).to(DEVICE)

    mid = generate_mid(model, frame_a, frame_b)
    mid_img = tensor_to_img(mid)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mid_img.save(out_path)
    print(f"Saved in-between frame to {out_path}")


if __name__ == "__main__":
    main()
