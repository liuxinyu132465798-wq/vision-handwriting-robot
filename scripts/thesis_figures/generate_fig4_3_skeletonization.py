import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


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


def preprocess_binary_for_skeleton(image_bgr: np.ndarray, auto_crop=True, exact_code_mode=True):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    clean, boxes = detect_valid_boxes(otsu)
    if not boxes:
        raise ValueError("没有检测到有效笔迹，请检查图片质量")

    if auto_crop:
        x0, y0, x1, y1 = get_crop_box(gray.shape, boxes, margin=15)
    else:
        x0, y0, x1, y1 = 0, 0, gray.shape[1], gray.shape[0]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_src = clean[y0:y1, x0:x1].copy()
    close_only = cv2.morphologyEx(close_src, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = close_only.copy()
    if exact_code_mode:
        morph = cv2.dilate(morph, kernel, iterations=1)

    extras = {
        "crop_box": (x0, y0, x1, y1),
        "original_crop": image_bgr[y0:y1, x0:x1].copy(),
        "gray_crop": gray[y0:y1, x0:x1].copy(),
        "blur_crop": blur[y0:y1, x0:x1].copy(),
        "otsu_crop": otsu[y0:y1, x0:x1].copy(),
        "close_only": close_only,
    }
    return morph, extras


def thinning_ximgproc(binary: np.ndarray):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)
    return None


def zhang_suen_thinning(binary: np.ndarray):
    # 输入要求：前景为255，背景为0
    img = (binary > 0).astype(np.uint8)
    prev = np.zeros_like(img)

    while True:
        marker = np.zeros_like(img)
        rows, cols = img.shape

        for step in (0, 1):
            marker.fill(0)
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    P1 = img[i, j]
                    if P1 != 1:
                        continue
                    P2 = img[i - 1, j]
                    P3 = img[i - 1, j + 1]
                    P4 = img[i, j + 1]
                    P5 = img[i + 1, j + 1]
                    P6 = img[i + 1, j]
                    P7 = img[i + 1, j - 1]
                    P8 = img[i, j - 1]
                    P9 = img[i - 1, j - 1]
                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]
                    B = sum(neighbors)
                    if B < 2 or B > 6:
                        continue
                    A = 0
                    seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
                    for k in range(8):
                        if seq[k] == 0 and seq[k + 1] == 1:
                            A += 1
                    if A != 1:
                        continue
                    if step == 0:
                        if P2 * P4 * P6 != 0:
                            continue
                        if P4 * P6 * P8 != 0:
                            continue
                    else:
                        if P2 * P4 * P8 != 0:
                            continue
                        if P2 * P6 * P8 != 0:
                            continue
                    marker[i, j] = 1
            img = img & (~marker)

        if np.array_equal(img, prev):
            break
        prev = img.copy()

    return (img * 255).astype(np.uint8)


def extract_skeleton(binary: np.ndarray):
    skel = thinning_ximgproc(binary)
    method = "cv2.ximgproc.thinning"
    if skel is None:
        skel = zhang_suen_thinning(binary)
        method = "内置 Zhang-Suen"
    return skel, method


def to_paper_style(img, invert=True):
    if img.ndim == 3:
        return img
    if invert:
        # 当前处理链路中前景是白色，论文图更适合白底黑线
        return 255 - img
    return img.copy()


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


def make_panel(img, title, panel_size=(420, 420), title_h=48):
    font = _find_font(24)
    canvas = Image.new("RGB", (panel_size[0], panel_size[1] + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_size[0], title_h), fill=(245, 245, 245))
    draw.text((12, 10), title, fill=(30, 30, 30), font=font)

    im = to_rgb_pil(img)
    im.thumbnail(panel_size, Image.Resampling.LANCZOS)
    x = (panel_size[0] - im.width) // 2
    y = title_h + (panel_size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def make_compare_figure(left_img, right_img, caption="图4-3 骨架提取效果示意图"):
    left = make_panel(left_img, "闭运算后的二值笔画图")
    right = make_panel(right_img, "Zhang-Suen细化后的单像素骨架图")

    gap = 20
    caption_h = 54
    w = left.width + right.width + gap
    h = max(left.height, right.height) + caption_h
    fig = Image.new("RGB", (w, h), (255, 255, 255))
    fig.paste(left, (0, 0))
    fig.paste(right, (left.width + gap, 0))

    draw = ImageDraw.Draw(fig)
    font = _find_font(24)
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    ty = max(left.height, right.height) + 10
    draw.text((tx, ty), caption, fill=(20, 20, 20), font=font)
    return fig


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("4.3 骨架提取效果图生成器")
        self.root.geometry("1180x840")

        self.image_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "exp43_output"))
        self.auto_crop = tk.BooleanVar(value=True)
        self.exact_code_mode = tk.BooleanVar(value=True)
        self.paper_style = tk.BooleanVar(value=True)
        self.save_individual = tk.BooleanVar(value=True)

        self.preview_photo = None
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="输入图片:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.image_path, width=95).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="选择图片", command=self.choose_image).grid(row=0, column=2)

        ttk.Label(top, text="输出目录:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(top, textvariable=self.output_dir, width=95).grid(row=1, column=1, padx=6, pady=(10, 0), sticky="ew")
        ttk.Button(top, text="选择目录", command=self.choose_output).grid(row=1, column=2, pady=(10, 0))

        opts = ttk.Frame(self.root, padding=(12, 0, 12, 0))
        opts.pack(fill="x")
        ttk.Checkbutton(opts, text="自动裁剪文字区域", variable=self.auto_crop).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opts, text="严格复现主程序（二值图=闭运算后再膨胀）", variable=self.exact_code_mode).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opts, text="论文显示模式（白底黑线）", variable=self.paper_style).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opts, text="同时保存单张图", variable=self.save_individual).pack(side="left")

        actions = ttk.Frame(self.root, padding=12)
        actions.pack(fill="x")
        ttk.Button(actions, text="生成并保存对比图", command=self.generate).pack(side="left")
        ttk.Button(actions, text="打开输出目录", command=self.open_output_dir).pack(side="left", padx=8)

        tip = ttk.Label(
            self.root,
            padding=(12, 0, 12, 8),
            foreground="#666666",
            text="说明：左图是闭运算后的二值笔画，右图是细化后的单像素骨架，用于说明笔画由宽线变为中心线。",
        )
        tip.pack(fill="x")

        preview_frame = ttk.LabelFrame(self.root, text="预览", padding=12)
        preview_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.preview_label = ttk.Label(preview_frame, text="请选择图片后点击“生成并保存对比图”")
        self.preview_label.pack(expand=True)

        self.log_box = tk.Text(self.root, height=10)
        self.log_box.pack(fill="x", padx=12, pady=(0, 12))

    def log(self, text):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.root.update_idletasks()

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("所有文件", "*.*")])
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

        out_dir = Path(self.output_dir.get().strip() or "exp43_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem

        try:
            image_bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("图片读取失败")

            binary_img, extras = preprocess_binary_for_skeleton(
                image_bgr,
                auto_crop=self.auto_crop.get(),
                exact_code_mode=self.exact_code_mode.get(),
            )
            skeleton, method = extract_skeleton(binary_img)

            left_show = to_paper_style(binary_img, invert=self.paper_style.get())
            right_show = to_paper_style(skeleton, invert=self.paper_style.get())
            fig = make_compare_figure(left_show, right_show)

            figure_path = out_dir / f"{stem}_图4-3_骨架提取效果示意图.png"
            fig.save(figure_path)
            self.log(f"✅ 已保存总对比图：{figure_path}")
            self.log(f"✅ 骨架提取方法：{method}")
            self.log(f"✅ 裁剪区域：{extras['crop_box']}")

            if self.save_individual.get():
                imwrite_unicode(str(out_dir / f"{stem}_01_闭运算后二值图.png"), left_show)
                imwrite_unicode(str(out_dir / f"{stem}_02_单像素骨架图.png"), right_show)
                self.log("✅ 单张图也已保存")

            preview = fig.copy()
            preview.thumbnail((1080, 620), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=self.preview_photo, text="")
            self.log("🎉 4.3 节骨架提取效果图生成完成")

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
