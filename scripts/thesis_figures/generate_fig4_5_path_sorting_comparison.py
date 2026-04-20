import math
import os
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None


# -------------------- 基础IO --------------------
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


# -------------------- 预处理（沿用主程序风格） --------------------
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


def preprocess_like_main_code(image_bgr: np.ndarray, auto_crop=True, exact_code_mode=True):
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
        "crop_box": (int(x0), int(y0), int(x1), int(y1)),
        "gray_crop": gray[y0:y1, x0:x1].copy(),
        "blur_crop": blur[y0:y1, x0:x1].copy(),
        "otsu_crop": otsu[y0:y1, x0:x1].copy(),
        "close_only": close_only,
    }
    return morph, extras


# -------------------- 骨架提取 --------------------
def thinning_ximgproc(binary: np.ndarray):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)
    return None


def zhang_suen_thinning(binary: np.ndarray):
    img = (binary > 0).astype(np.uint8)
    prev = np.zeros_like(img)

    while True:
        rows, cols = img.shape
        for step in (0, 1):
            marker = np.zeros_like(img)
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if img[i, j] != 1:
                        continue
                    p2 = img[i - 1, j]
                    p3 = img[i - 1, j + 1]
                    p4 = img[i, j + 1]
                    p5 = img[i + 1, j + 1]
                    p6 = img[i + 1, j]
                    p7 = img[i + 1, j - 1]
                    p8 = img[i, j - 1]
                    p9 = img[i - 1, j - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    B = sum(neighbors)
                    if B < 2 or B > 6:
                        continue
                    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                    A = sum(1 for k in range(8) if seq[k] == 0 and seq[k + 1] == 1)
                    if A != 1:
                        continue
                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
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


# -------------------- 路径提取与排序（按当前主程序） --------------------
def skeleton_to_paths(skel: np.ndarray, min_len=6):
    points = set(map(tuple, np.argwhere(skel > 0)))
    if not points:
        return []

    neigh_cache = {}

    def neighbors(p):
        if p in neigh_cache:
            return neigh_cache[p]
        y, x = p
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in points:
                    out.append(q)
        neigh_cache[p] = out
        return out

    degree = {p: len(neighbors(p)) for p in points}
    nodes = {p for p, d in degree.items() if d != 2}

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    visited_edges = set()
    paths = []

    for node in list(nodes):
        for nb in neighbors(node):
            key = edge_key(node, nb)
            if key in visited_edges:
                continue
            path = [node, nb]
            visited_edges.add(key)
            prev, cur = node, nb
            while cur not in nodes:
                nbrs = [n for n in neighbors(cur) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                key = edge_key(cur, nxt)
                if key in visited_edges:
                    break
                path.append(nxt)
                visited_edges.add(key)
                prev, cur = cur, nxt
            if len(path) >= min_len:
                paths.append([(p[1], p[0]) for p in path])

    seen_loop = set()
    for start in [p for p in points if degree[p] == 2]:
        if start in seen_loop:
            continue
        loop = [start]
        seen_loop.add(start)
        prev = None
        cur = start
        while True:
            nbrs = [n for n in neighbors(cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start or nxt in seen_loop:
                break
            loop.append(nxt)
            seen_loop.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= min_len:
            paths.append([(p[1], p[0]) for p in loop])
    return paths


def smooth_path(path):
    if len(path) < 7 or savgol_filter is None:
        return path
    pts = np.array(path, dtype=np.float32)
    win = min(11, len(path) if len(path) % 2 == 1 else len(path) - 1)
    if win < 5:
        return path
    try:
        x = savgol_filter(pts[:, 0], win, 2, mode="interp")
        y = savgol_filter(pts[:, 1], win, 2, mode="interp")
        return list(zip(x, y))
    except Exception:
        return path


def resample_path(path, step=1.2):
    if len(path) < 2:
        return path
    pts = np.array(path, dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < step:
        return path
    s = np.concatenate([[0], np.cumsum(seg)])
    new_s = np.arange(0, s[-1] + 1e-6, step)
    xs = np.interp(new_s, s, pts[:, 0])
    ys = np.interp(new_s, s, pts[:, 1])
    return list(zip(xs, ys))


def clean_path(path, min_dist=0.8):
    if len(path) < 2:
        return []
    cleaned = [path[0]]
    for p in path[1:]:
        if math.dist(p, cleaned[-1]) > min_dist:
            cleaned.append(p)
    return cleaned


def sort_paths(polys):
    if len(polys) <= 1:
        return polys
    remaining = [np.array(p, dtype=np.float32) for p in polys]
    out = [remaining.pop(0)]
    current_end = out[0][-1]

    while remaining:
        best_i = None
        best_reverse = False
        best_dist = 1e18

        for i, p in enumerate(remaining):
            d0 = np.linalg.norm(p[0] - current_end)
            d1 = np.linalg.norm(p[-1] - current_end)
            if d0 < best_dist:
                best_dist = d0
                best_i = i
                best_reverse = False
            if d1 < best_dist:
                best_dist = d1
                best_i = i
                best_reverse = True

        best = remaining.pop(best_i)
        if best_reverse:
            best = best[::-1]
        out.append(best)
        current_end = best[-1]

    return [p.tolist() for p in out]


def extract_polys_from_image(input_path, auto_crop=True, exact_code_mode=True):
    image_bgr = imread_unicode(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("图片读取失败")

    binary_roi, extras = preprocess_like_main_code(image_bgr, auto_crop=auto_crop, exact_code_mode=exact_code_mode)
    skel, skel_method = extract_skeleton(binary_roi)

    raw_paths = skeleton_to_paths(skel, min_len=6)
    polys = []
    for path in raw_paths:
        path = smooth_path(path)
        path = resample_path(path, step=1.2)
        path = clean_path(path, min_dist=0.8)
        if len(path) >= 6:
            polys.append(path)

    if not polys:
        raise ValueError("没有提取到有效路径")

    info = {
        "crop_box": extras["crop_box"],
        "skeleton_method": skel_method,
        "binary_roi": binary_roi,
        "skeleton": skel,
        "raw_path_count": len(raw_paths),
        "poly_count": len(polys),
    }
    return polys, info


# -------------------- 可视化 --------------------
def _find_font(font_size=24):
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
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


def draw_dashed_line(draw, p1, p2, dash=10, gap=8, width=2, fill=(220, 0, 0)):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return
    ux = dx / dist
    uy = dy / dist
    pos = 0.0
    while pos < dist:
        s = pos
        e = min(pos + dash, dist)
        a = (x1 + ux * s, y1 + uy * s)
        b = (x1 + ux * e, y1 + uy * e)
        draw.line([a, b], fill=fill, width=width)
        pos += dash + gap


def render_motion_panel(paths, title, bg_shape, line_width=3, dash_width=2):
    h, w = bg_shape[:2]
    panel = Image.new("RGB", (w, h + 66), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font = _find_font(24)
    small = _find_font(18)
    draw.rectangle((0, 0, w, 54), fill=(245, 245, 245))
    draw.text((14, 12), title, fill=(20, 20, 20), font=font)

    offset_y = 60
    draw.rectangle((0, offset_y, w - 1, offset_y + h - 1), outline=(220, 220, 220), width=1)

    penup_len = 0.0
    penup_count = 0
    writing_len = 0.0

    for idx, path in enumerate(paths):
        if len(path) >= 2:
            for a, b in zip(path, path[1:]):
                writing_len += math.dist(a, b)
            draw.line([(float(x), float(y + offset_y)) for x, y in path], fill=(0, 0, 0), width=line_width, joint="curve")
        if idx < len(paths) - 1 and len(path) >= 1 and len(paths[idx + 1]) >= 1:
            a = path[-1]
            b = paths[idx + 1][0]
            penup_len += math.dist(a, b)
            penup_count += 1
            draw_dashed_line(
                draw,
                (float(a[0]), float(a[1] + offset_y)),
                (float(b[0]), float(b[1] + offset_y)),
                dash=10,
                gap=8,
                width=dash_width,
                fill=(220, 0, 0),
            )

    stat_text = f"书写段: {len(paths)} 条    空行程: {penup_count} 次    空行程总长: {penup_len:.1f}px"
    draw.text((14, h + offset_y + 6), stat_text, fill=(70, 70, 70), font=small)
    return panel, {"penup_len": penup_len, "penup_count": penup_count, "writing_len": writing_len}


def make_compare_figure(before_panel, after_panel, caption="图4-5 路径排序优化前后对比图"):
    gap = 24
    caption_h = 56
    w = before_panel.width + after_panel.width + gap
    h = max(before_panel.height, after_panel.height) + caption_h
    fig = Image.new("RGB", (w, h), (255, 255, 255))
    fig.paste(before_panel, (0, 0))
    fig.paste(after_panel, (before_panel.width + gap, 0))

    draw = ImageDraw.Draw(fig)
    font = _find_font(24)
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    ty = max(before_panel.height, after_panel.height) + 12
    draw.text((tx, ty), caption, fill=(20, 20, 20), font=font)
    return fig


def make_legend_image():
    w, h = 460, 70
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _find_font(20)
    draw.line([(24, 24), (92, 24)], fill=(0, 0, 0), width=3)
    draw.text((108, 12), "黑色：实际书写段", fill=(0, 0, 0), font=font)
    draw_dashed_line(draw, (24, 50), (92, 50), dash=10, gap=8, width=2, fill=(220, 0, 0))
    draw.text((108, 38), "红色虚线：抬笔空行程", fill=(180, 0, 0), font=font)
    return img


# -------------------- 核心流程 --------------------
def generate_sort_compare_figure(
    input_path,
    outdir,
    auto_crop=True,
    exact_code_mode=True,
    demo_shuffle_before=False,
    random_seed=42,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    polys, base_info = extract_polys_from_image(input_path, auto_crop=auto_crop, exact_code_mode=exact_code_mode)

    before_paths = [list(map(tuple, p)) for p in polys]
    if demo_shuffle_before and len(before_paths) >= 3:
        rng = random.Random(random_seed)
        rng.shuffle(before_paths)

    after_paths = [list(map(tuple, p)) for p in sort_paths(before_paths)]

    h = int(max(max(y for p in before_paths for _, y in p), max(y for p in after_paths for _, y in p)) + 20)
    w = int(max(max(x for p in before_paths for x, _ in p), max(x for p in after_paths for x, _ in p)) + 20)
    bg_shape = (h, w)

    before_panel, before_stat = render_motion_panel(before_paths, "排序前", bg_shape)
    after_panel, after_stat = render_motion_panel(after_paths, "排序后", bg_shape)
    fig = make_compare_figure(before_panel, after_panel)
    legend = make_legend_image()

    before_path = outdir / f"{stem}_01_排序前运动路径.png"
    after_path = outdir / f"{stem}_02_排序后运动路径.png"
    fig_path = outdir / f"{stem}_图4-5_路径排序优化前后对比图.png"
    legend_path = outdir / f"{stem}_图例_黑色书写段_红色虚线空行程.png"

    before_panel.save(before_path)
    after_panel.save(after_path)
    fig.save(fig_path)
    legend.save(legend_path)

    info = {
        "before_path": str(before_path),
        "after_path": str(after_path),
        "fig_path": str(fig_path),
        "legend_path": str(legend_path),
        "crop_box": base_info["crop_box"],
        "skeleton_method": base_info["skeleton_method"],
        "raw_path_count": base_info["raw_path_count"],
        "poly_count": base_info["poly_count"],
        "before_penup_len": before_stat["penup_len"],
        "after_penup_len": after_stat["penup_len"],
        "before_penup_count": before_stat["penup_count"],
        "after_penup_count": after_stat["penup_count"],
        "improvement_ratio": 0.0 if before_stat["penup_len"] < 1e-9 else (before_stat["penup_len"] - after_stat["penup_len"]) / before_stat["penup_len"],
        "demo_shuffle_before": bool(demo_shuffle_before),
    }
    return fig, legend, info


# -------------------- GUI --------------------
class Exp45GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("4.5 路径排序优化前后对比图生成器")
        self.root.geometry("1220x900")

        self.input_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value=str(Path.cwd() / "exp45_output"))
        self.auto_crop_var = tk.BooleanVar(value=True)
        self.exact_code_var = tk.BooleanVar(value=True)
        self.demo_shuffle_var = tk.BooleanVar(value=False)
        self.seed_var = tk.IntVar(value=42)

        self.preview_label = None
        self._preview_imgtk = None

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="输入图片:").pack(side="left")
        ttk.Entry(row1, textvariable=self.input_var, width=92).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row1, text="选择图片", command=self.choose_input).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="输出目录:").pack(side="left")
        ttk.Entry(row2, textvariable=self.outdir_var, width=90).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row2, text="选择目录", command=self.choose_outdir).pack(side="left")

        opts = ttk.LabelFrame(top, text="参数设置", padding=10)
        opts.pack(fill="x", pady=8)

        row3 = ttk.Frame(opts)
        row3.pack(fill="x", pady=4)
        ttk.Checkbutton(row3, text="自动裁剪文字区域", variable=self.auto_crop_var).pack(side="left", padx=4)
        ttk.Checkbutton(row3, text="完全沿用主程序形态学（闭运算+膨胀）", variable=self.exact_code_var).pack(side="left", padx=12)

        row4 = ttk.Frame(opts)
        row4.pack(fill="x", pady=4)
        ttk.Checkbutton(row4, text="演示增强：打乱排序前路径（仅用于让差异更明显）", variable=self.demo_shuffle_var).pack(side="left", padx=4)
        ttk.Label(row4, text="随机种子:").pack(side="left", padx=(16, 0))
        ttk.Entry(row4, textvariable=self.seed_var, width=8).pack(side="left", padx=6)
        ttk.Label(row4, text="正常论文图建议关闭；只有差异不明显时再打开。", foreground="#666666").pack(side="left", padx=10)

        row5 = ttk.Frame(top)
        row5.pack(fill="x", pady=8)
        ttk.Button(row5, text="生成图4-5", command=self.run).pack(side="left", padx=4)
        ttk.Button(row5, text="打开输出目录", command=self.open_outdir_hint).pack(side="left", padx=4)

        self.info_text = tk.Text(self.root, height=10)
        self.info_text.pack(fill="x", padx=10, pady=6)

        preview_frame = ttk.LabelFrame(self.root, text="预览", padding=10)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(expand=True)

    def choose_input(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.input_var.set(path)

    def choose_outdir(self):
        path = filedialog.askdirectory()
        if path:
            self.outdir_var.set(path)

    def open_outdir_hint(self):
        outdir = Path(self.outdir_var.get())
        outdir.mkdir(parents=True, exist_ok=True)
        messagebox.showinfo("输出目录", f"输出目录已准备好：\n{outdir}")

    def log(self, msg):
        self.info_text.insert("end", msg + "\n")
        self.info_text.see("end")
        self.root.update_idletasks()

    def show_preview(self, pil_img: Image.Image):
        view = pil_img.copy()
        view.thumbnail((1100, 620), Image.Resampling.LANCZOS)
        self._preview_imgtk = ImageTk.PhotoImage(view)
        self.preview_label.configure(image=self._preview_imgtk)

    def run(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("提示", "请先选择输入图片")
            return
        try:
            self.info_text.delete("1.0", "end")
            fig, legend, info = generate_sort_compare_figure(
                input_path=input_path,
                outdir=self.outdir_var.get().strip() or "exp45_output",
                auto_crop=self.auto_crop_var.get(),
                exact_code_mode=self.exact_code_var.get(),
                demo_shuffle_before=self.demo_shuffle_var.get(),
                random_seed=int(self.seed_var.get()),
            )
            self.show_preview(fig)
            self.log(f"已生成总图: {info['fig_path']}")
            self.log(f"图例: {info['legend_path']}")
            self.log(f"裁剪区域: {info['crop_box']}")
            self.log(f"骨架方法: {info['skeleton_method']}")
            self.log(f"初始有效路径数: {info['poly_count']}")
            self.log(f"排序前空行程: {info['before_penup_count']} 次, 总长 {info['before_penup_len']:.1f}px")
            self.log(f"排序后空行程: {info['after_penup_count']} 次, 总长 {info['after_penup_len']:.1f}px")
            self.log(f"空行程长度下降比例: {info['improvement_ratio'] * 100:.1f}%")
            if info['demo_shuffle_before']:
                self.log("注意：当前开启了“演示增强”，排序前路径顺序已被打乱，仅适合做示意图。")
            messagebox.showinfo("完成", f"图4-5已生成：\n{info['fig_path']}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    Exp45GUI().mainloop()
