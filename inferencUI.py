import sys
import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

# -------------------------
# CONFIG
# -------------------------
CHECKPOINT_PATH = "checkpoints/model_epoch_2.pt"
IMAGE_SIZE = 128
DIFFUSION_STEPS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# IMAGE UTILS
# -------------------------
def img_to_tensor(img):
    arr = np.array(img).astype(np.float32)
    rgb = arr[..., :3] / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1)

def tensor_to_img(t):
    t = (t.clamp(-1, 1) + 1) * 127.5
    arr = t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr, "RGB")

# -------------------------
# PALETTE LOCKING
# -------------------------
def extract_palette(images, max_colors=32):
    pixels = []
    for img in images:
        arr = np.array(img)
        rgb = arr[..., :3]
        pixels.append(rgb.reshape(-1, 3))
    pixels = np.vstack(pixels)

    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    idx = np.argsort(-counts)[:max_colors]
    return colors[idx]

def lock_palette(img, palette):
    arr = np.array(img)
    flat = arr.reshape(-1, 3).astype(np.int16)
    pal = palette.astype(np.int16)

    dists = ((flat[:, None] - pal[None]) ** 2).sum(axis=2)
    nearest = pal[dists.argmin(axis=1)]

    locked = nearest.reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(locked, "RGB")

# -------------------------
# MODEL LOADING
# -------------------------
def load_model():
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=9,
        out_channels=3,
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

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------
# DIFFUSION INFERENCE
# -------------------------
@torch.no_grad()
def generate_inbetween(model, frame_a, frame_b):
    scheduler = DDPMScheduler(
        num_train_timesteps=DIFFUSION_STEPS,
        beta_schedule="squaredcos_cap_v2"
    )

    x = torch.randn_like(frame_a).to(DEVICE)

    for t in reversed(range(DIFFUSION_STEPS)):
        t_tensor = torch.tensor([t], device=DEVICE)
        model_input = torch.cat([x, frame_a, frame_b], dim=1)
        noise_pred = model(model_input, t_tensor).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x

# -------------------------
# QT APP
# -------------------------
class InbetweenApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel In-between Generator")

        self.model = load_model()

        self.img_a = None
        self.img_b = None

        self.label_a = QLabel("Frame A")
        self.label_mid = QLabel("In-between")
        self.label_b = QLabel("Frame B")

        for lbl in (self.label_a, self.label_mid, self.label_b):
            lbl.setFixedSize(IMAGE_SIZE * 2, IMAGE_SIZE * 2)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("border: 1px solid gray;")

        load_a_btn = QPushButton("Load Frame A")
        gen_btn = QPushButton("Generate In-between")
        load_b_btn = QPushButton("Load Frame B")

        load_a_btn.clicked.connect(self.load_a)
        load_b_btn.clicked.connect(self.load_b)
        gen_btn.clicked.connect(self.generate)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.label_a)
        img_layout.addWidget(self.label_mid)
        img_layout.addWidget(self.label_b)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_a_btn)
        btn_layout.addWidget(gen_btn)
        btn_layout.addWidget(load_b_btn)

        layout = QVBoxLayout()
        layout.addLayout(img_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_a(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Frame A")
        if path:
            self.img_a = Image.open(path).convert("RGB")
            self.label_a.setPixmap(self.pil_to_pixmap(self.img_a))
            # self.img_a = Image.open(path).convert("RGB")
            # self.label_a.setPixmap(QPixmap.fromImage(self.img_a))
            # data = self.img_a.tobytes("raw","RGB")
            # qim = QImage(data, self.img_a.size[0], self.img_a.size[1], QtGui.QImage.Format_ARGB32)
            # pix = QPixmap.fromImage(qim)

    def load_b(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Frame B")
        if path:
            self.img_b = Image.open(path).convert("RGB")
            self.label_b.setPixmap(self.pil_to_pixmap(self.img_b))

    def generate(self):
        if self.img_a is None or self.img_b is None:
            return

        frame_a = img_to_tensor(self.img_a).unsqueeze(0).to(DEVICE)
        frame_b = img_to_tensor(self.img_b).unsqueeze(0).to(DEVICE)

        mid = generate_inbetween(self.model, frame_a, frame_b)
        mid_img = tensor_to_img(mid[0])

        palette = extract_palette([self.img_a, self.img_b])
        mid_locked = lock_palette(mid_img, palette)

        self.label_mid.setPixmap(self.pil_to_pixmap(mid_locked))

    def pil_to_pixmap(self, img):
        img = img.resize((IMAGE_SIZE * 2, IMAGE_SIZE * 2), Image.NEAREST)
        img = img.convert("RGB")
        w, h = img.size
        data = img.tobytes("raw", "RGB")
        print("all good")

        qimg = QImage(
            data,
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888
        )

        print("all good 2")

        return QPixmap.fromImage(qimg)

# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    
    import multiprocessing
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    window = InbetweenApp()
    window.show()
    sys.exit(app.exec())
