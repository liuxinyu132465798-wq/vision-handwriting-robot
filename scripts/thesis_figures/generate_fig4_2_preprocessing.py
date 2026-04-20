import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


# =========================
#  与主程序保持一致的核心预处理流程
# =========================

def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def detect_valid_boxes(binary: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 20 and h >= 8 and w >= 2:
            clean[labels == i] = 255
            boxes.append((x, y, w, h))
    return clean, boxes


def get_crop_box(img_shape, boxes, margin=15):
    h, w = img_shape[:2]
    x0 = max(0, min(b[0] for b in boxes) - margin)
    y0 = max(0, min(b[1] for b in boxes) - margin)
    x1 = min(w, max(b[0] + b[2] for b in boxes) + margin)
    y1 = min(h, max(b[1] + b[3] for b in boxes) + margin)
    return x0, y0, x1, y1


def preprocess_pipeline(image_bgr: np.ndarray, auto_crop=True, exact_code_mode=True):
    if image_bgr is None:
        raise ValueError("图片读取失败")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 与你的主程序 preprocess_image() 保持一致：
    # 大尺度高斯背景 -> 除法归一化 -> 再做 3x3 高斯平滑 -> Otsu
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    clean, boxes = detect_valid_boxes(otsu)
    if not boxes:
        raise ValueError("没有检测到有效笔迹，请检查拍照亮度、对焦和纸张背景")

    if auto_crop:
        x0, y0, x1, y1 = get_crop_box(gray.shape, boxes, margin=15)
    else:
        x0, y0, x1, y1 = 0, 0, gray.shape[1], gray.shape[0]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_src = clean[y0:y1, x0:x1].copy()
    close_only = cv2.morphologyEx(close_src, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = close_only.copy()

    # 主程序里闭运算后还做了一次膨胀，这里提供“严格复现模式”
    if exact_code_mode:
        morph = cv2.dilate(morph, kernel, iterations=1)

    stages = {
        "原始图像": image_bgr[y0:y1, x0:x1].copy(),
        "灰度图": gray[y0:y1, x0:x1].copy(),
        "高斯模糊图": blur[y0:y1, x0:x1].copy(),
        "Otsu二值图": otsu[y0:y1, x0:x1].copy(),
        "闭运算结果图": morph,
    }

    extras = {
        "背景均衡图": norm[y0:y1, x0:x1].copy(),
        "连通域去噪图": clean[y0:y1, x0:x1].copy(),
        "仅闭运算": close_only,
        "crop_box": (x0, y0, x1, y1),
    }
    return stages, extras


def _find_font(font_size=24):
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()


def to_rgb_pil(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def make_panel(img, title, panel_size=(320, 320), title_h=42, bg=(255, 255, 255)):
    font = _find_font(22)
    canvas = Image.new("RGB", (panel_size[0], panel_size[1] + title_h), bg)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_size[0], title_h), fill=(245, 245, 245))
    draw.text((12, 8), title, fill=(30, 30, 30), font=font)

    im = to_rgb_pil(img)
    im.thumbnail(panel_size, Image.Resampling.LANCZOS)
    x = (panel_size[0] - im.width) // 2
    y = title_h + (panel_size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def make_compare_figure(stages: dict, panel_size=(320, 320), gap=16):
    order = ["原始图像", "灰度图", "高斯模糊图", "Otsu二值图", "闭运算结果图"]
    panels = [make_panel(stages[k], k, panel_size=panel_size) for k in order]
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    figure = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        figure.paste(p, (x, 0))
        x += p.width + gap
    return figure


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("4.2 图像预处理过程对比图生成器")
        self.root.geometry("1160x820")

        self.image_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "exp42_output"))
        self.auto_crop = tk.BooleanVar(value=True)
        self.exact_code_mode = tk.BooleanVar(value=True)
        self.save_individual = tk.BooleanVar(value=True)

        self.preview_photo = None
        self.last_figure_path = None

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="输入图片:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.image_path, width=95).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="选择图片", command=self.choose_image).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="输出目录:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.output_dir, width=95).grid(row=1, column=1, padx=6, pady=(10, 0), sticky="ew")
        ttk.Button(top, text="选择目录", command=self.choose_output).grid(row=1, column=2, padx=6, pady=(10, 0))

        options = ttk.Frame(self.root, padding=(12, 0, 12, 0))
        options.pack(fill="x")
        ttk.Checkbutton(options, text="自动裁剪文字区域（更适合论文插图）", variable=self.auto_crop).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(options, text="严格复现主程序（闭运算后再膨胀）", variable=self.exact_code_mode).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(options, text="同时保存单张过程图", variable=self.save_individual).pack(side="left")

        actions = ttk.Frame(self.root, padding=12)
        actions.pack(fill="x")
        ttk.Button(actions, text="生成并保存对比图", command=self.generate).pack(side="left")
        ttk.Button(actions, text="打开输出目录", command=self.open_output_dir).pack(side="left", padx=8)

        tip = ttk.Label(
            self.root,
            padding=(12, 0, 12, 8),
            foreground="#666666",
            text="说明：本程序按你主代码的预处理核心流程生成 5 张小图，可直接作为 4.2 节实验过程图。",
        )
        tip.pack(fill="x")

        preview_frame = ttk.LabelFrame(self.root, text="预览", padding=12)
        preview_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.preview_label = ttk.Label(preview_frame, text="请选择图片后点击“生成并保存对比图”")
        self.preview_label.pack(expand=True)

        self.log_box = tk.Text(self.root, height=9)
        self.log_box.pack(fill="x", padx=12, pady=(0, 12))

    def log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.root.update_idletasks()

    def choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        if path:
            self.image_path.set(path)

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def open_output_dir(self):
        out = Path(self.output_dir.get())
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out))  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("提示", f"输出目录：{out}")

    def generate(self):
        img_path = self.image_path.get().strip()
        if not img_path:
            messagebox.showwarning("提示", "请先选择图片")
            return

        out_dir = Path(self.output_dir.get().strip() or "exp42_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem

        try:
            image_bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
            stages, extras = preprocess_pipeline(
                image_bgr,
                auto_crop=self.auto_crop.get(),
                exact_code_mode=self.exact_code_mode.get(),
            )
            figure = make_compare_figure(stages, panel_size=(320, 320), gap=16)

            figure_path = out_dir / f"{stem}_4.2_预处理对比图.png"
            figure.save(figure_path)
            self.last_figure_path = str(figure_path)
            self.log(f"✅ 已保存总对比图：{figure_path}")
            self.log(f"✅ 裁剪区域：{extras['crop_box']}")

            if self.save_individual.get():
                name_map = {
                    "原始图像": "01_原始图像.png",
                    "灰度图": "02_灰度图.png",
                    "高斯模糊图": "03_高斯模糊图.png",
                    "Otsu二值图": "04_Otsu二值图.png",
                    "闭运算结果图": "05_闭运算结果图.png",
                }
                for k, v in stages.items():
                    save_path = out_dir / f"{stem}_{name_map[k]}"
                    imwrite_unicode(str(save_path), v)
                imwrite_unicode(str(out_dir / f"{stem}_06_背景均衡图.png"), extras["背景均衡图"])
                imwrite_unicode(str(out_dir / f"{stem}_07_连通域去噪图.png"), extras["连通域去噪图"])
                imwrite_unicode(str(out_dir / f"{stem}_08_仅闭运算.png"), extras["仅闭运算"])
                self.log("✅ 单张过程图也已保存（含背景均衡图、连通域去噪图等扩展图）")

            preview = figure.copy()
            preview.thumbnail((1080, 560), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_photo, text="")
            self.log("🎉 4.2 节实验过程图生成完成，可直接插入论文")

        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.log(f"❌ 生成失败：{e}")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()
