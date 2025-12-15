import sys
from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QSpinBox, QListWidget, QMessageBox
)
from PySide6.QtGui import QPixmap, QPen
from PySide6.QtCore import Qt, QRectF

TARGET_SIZE = 128

class SpriteExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Sprite Dataset Extractor (Grid Mode)")

        self.image = None
        self.frames = []

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Controls
        self.load_btn = QPushButton("Load Sprite Sheet")
        self.preview_btn = QPushButton("Preview Grid")
        self.extract_btn = QPushButton("Extract Frames")
        self.export_btn = QPushButton("Export Triplets")

        self.rows = QSpinBox()
        self.cols = QSpinBox()
        self.sx = QSpinBox()
        self.sy = QSpinBox()

        for s in (self.rows, self.cols):
            s.setRange(1, 256)
            s.setValue(4)

        for s in (self.sx, self.sy):
            s.setRange(0, 64)
            s.setValue(0)

        self.frame_list = QListWidget()

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Rows"))
        controls.addWidget(self.rows)
        controls.addWidget(QLabel("Cols"))
        controls.addWidget(self.cols)
        controls.addWidget(QLabel("Spacing X"))
        controls.addWidget(self.sx)
        controls.addWidget(QLabel("Spacing Y"))
        controls.addWidget(self.sy)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(controls)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.preview_btn)
        layout.addWidget(self.extract_btn)
        layout.addWidget(self.frame_list)
        layout.addWidget(self.export_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Signals
        self.load_btn.clicked.connect(self.load_image)
        self.preview_btn.clicked.connect(self.preview_grid)
        self.extract_btn.clicked.connect(self.extract_frames)
        self.export_btn.clicked.connect(self.export_triplets)

    # -------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Sprite Sheet", "", "PNG Files (*.png)")
        if not path:
            return

        self.image = Image.open(path).convert("RGBA")
        self.scene.clear()
        self.scene.addPixmap(QPixmap(path))
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.frames.clear()
        self.frame_list.clear()

    # -------------------------
    def compute_frame_size(self):
        iw, ih = self.image.size
        rows, cols = self.rows.value(), self.cols.value()
        sx, sy = self.sx.value(), self.sy.value()

        fw = (iw - sx * (cols - 1)) // cols
        fh = (ih - sy * (rows - 1)) // rows

        if fw <= 0 or fh <= 0:
            raise ValueError("Invalid grid parameters")

        return fw, fh

    # -------------------------
    def preview_grid(self):
        if self.image is None:
            return

        fw, fh = self.compute_frame_size()
        rows, cols = self.rows.value(), self.cols.value()
        sx, sy = self.sx.value(), self.sy.value()

        # Clear old grid
        for item in self.scene.items():
            if isinstance(item, QGraphicsRectItem):
                self.scene.removeItem(item)

        pen = QPen(Qt.green)
        pen.setWidth(1)

        y = 0
        for r in range(rows):
            x = 0
            for c in range(cols):
                rect = QGraphicsRectItem(QRectF(x, y, fw, fh))
                rect.setPen(pen)
                self.scene.addItem(rect)
                x += fw + sx
            y += fh + sy

    # -------------------------
    def extract_frames(self):
        if self.image is None:
            return

        fw, fh = self.compute_frame_size()
        rows, cols = self.rows.value(), self.cols.value()
        sx, sy = self.sx.value(), self.sy.value()

        self.frames.clear()
        self.frame_list.clear()

        for r in range(rows):
            for c in range(cols):
                x = c * (fw + sx)
                y = r * (fh + sy)
                crop = self.image.crop((x, y, x + fw, y + fh))
                if crop.getbbox():
                    self.frames.append(self.pad(crop))
                    self.frame_list.addItem(f"Frame {len(self.frames)}")

        QMessageBox.information(self, "Done", f"Extracted {len(self.frames)} frames.")

    # -------------------------
    def pad(self, img):
        canvas = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
        w, h = img.size
        canvas.paste(img, ((TARGET_SIZE - w) // 2, (TARGET_SIZE - h) // 2))
        return canvas

    # -------------------------
    def export_triplets(self):
        if len(self.frames) < 3:
            return

        out = QFileDialog.getExistingDirectory(self, "Output Folder")
        if not out:
            return

        out = Path(out)
        idx = 0

        for i in range(len(self.frames) - 2):
            d = out / f"triplet_{idx:04d}"
            d.mkdir(parents=True, exist_ok=True)
            self.frames[i].save(d / "A.png")
            self.frames[i + 1].save(d / "mid.png")
            self.frames[i + 2].save(d / "B.png")
            idx += 1

        QMessageBox.information(self, "Exported", f"{idx} triplets created.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SpriteExtractor()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
