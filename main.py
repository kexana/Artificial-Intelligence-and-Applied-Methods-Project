import os
import math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator

# -------------------------
# CONFIG
# -------------------------
DATASET_ROOT = "dataset"
OUTPUT_DIR = "checkpoints"
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 2 #50
LR = 2e-4
DIFFUSION_STEPS = 12   # VERY IMPORTANT for pixel art
SAVE_EVERY = 5
USE_RGBA = False      # RGB is usually enough

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# UTILITIES
# -------------------------
def img_to_tensor(img):
    arr = np.array(img).astype(np.float32)
    if USE_RGBA:
        rgb = arr[..., :4] / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1)
    else:
        rgb = arr[..., :3] / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1)


# -------------------------
# DATASET
# -------------------------
class SpriteTripletDataset(Dataset):
    def __init__(self, root):
        self.triplets = []
        root = Path(root)

        for triplet in root.glob("**/triplet_*"):
            A = triplet / "A.png"
            M = triplet / "mid.png"
            B = triplet / "B.png"
            if A.exists() and M.exists() and B.exists():
                self.triplets.append((A, M, B))

        if len(self.triplets) == 0:
            raise RuntimeError("No triplets found!")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        A_path, M_path, B_path = self.triplets[idx]

        A = Image.open(A_path).convert("RGBA")
        M = Image.open(M_path).convert("RGBA")
        B = Image.open(B_path).convert("RGBA")

        A = img_to_tensor(A)
        M = img_to_tensor(M)
        B = img_to_tensor(B)

        return {
            "frame_a": A,
            "frame_mid": M,
            "frame_b": B
        }

def main():
    # -------------------------
    # MODEL
    # -------------------------
    IN_CHANNELS = 9 if not USE_RGBA else 12
    OUT_CHANNELS = 3 if not USE_RGBA else 4

    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
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

    # -------------------------
    # DIFFUSION
    # -------------------------
    scheduler = DDPMScheduler(
        num_train_timesteps=DIFFUSION_STEPS,
        beta_schedule="squaredcos_cap_v2"
    )

    # -------------------------
    # TRAIN SETUP
    # -------------------------
    dataset = SpriteTripletDataset(DATASET_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    accelerator = Accelerator()
    model, optimizer, loader = accelerator.prepare(
        model, optimizer, loader
    )

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in progress:
            frame_a = batch["frame_a"]
            frame_b = batch["frame_b"]
            target = batch["frame_mid"]

            noise = torch.randn_like(target)
            timesteps = torch.randint(
                0, DIFFUSION_STEPS, (target.shape[0],),
                device=target.device
            )

            noisy_target = scheduler.add_noise(target, noise, timesteps)

            model_input = torch.cat(
                [noisy_target, frame_a, frame_b], dim=1
            )

            pred_noise = model(model_input, timesteps).sample

            loss = F.mse_loss(pred_noise, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress.set_postfix(loss=loss.item())

        # -------------------------
        # SAVE CHECKPOINT
        # -------------------------
        if (epoch + 1) % SAVE_EVERY == 0:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)

            ckpt_path = Path(OUTPUT_DIR) / f"model_epoch_{epoch+1}.pt"
            torch.save(unwrapped.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
