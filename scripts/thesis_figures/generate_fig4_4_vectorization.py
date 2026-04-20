import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


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


# -------------------- 沿用主程序的预处理 --------------------
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
        "crop_box": (x0, y0, x1, y1),
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


# -------------------- 当前代码式路径提取 --------------------
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


# -------------------- Suzuki / Douglas --------------------
def find_contours_suzuki(src: np.ndarray, retrieval_mode="list", min_points=8):
    mode_map = {
        "list": cv2.RETR_LIST,
        "external": cv2.RETR_EXTERNAL,
        "tree": cv2.RETR_TREE,
    }
    mode = mode_map.get(retrieval_mode, cv2.RETR_LIST)
    contours, _ = cv2.findContours(src.copy(), mode, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if c is None or len(c) < min_points:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        out.append(pts)
    out.sort(key=lambda a: (float(np.min(a[:, 0])), float(np.min(a[:, 1]))))
    return out


def simplify_polyline_dp(points: np.ndarray, epsilon_px=2.0, closed=False):
    if len(points) < 3:
        return points.copy()
    curve = points.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(curve, epsilon_px, closed)
    return approx.reshape(-1, 2).astype(np.float32)


# -------------------- 绘图 --------------------
def to_paper_style(img, invert=True):
    if img.ndim == 3:
        if invert:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return 255 - gray
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (255 - img) if invert else img.copy()


def blank_paper_like(gray: np.ndarray):
    h, w = gray.shape[:2]
    return np.full((h, w, 3), 255, dtype=np.uint8)


def draw_paths_canvas(base_gray: np.ndarray, paths, line_thickness=1, point_radius=0, connect=True):
    canvas = blank_paper_like(base_gray)
    if base_gray.ndim == 2:
        faint = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        faint = cv2.addWeighted(canvas, 0.88, faint, 0.12, 0)
        canvas = faint

    for pts in paths:
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) == 0:
            continue
        ipts = np.round(pts).astype(np.int32)
        if connect and len(ipts) >= 2:
            cv2.polylines(canvas, [ipts.reshape(-1, 1, 2)], False, (0, 0, 0), line_thickness, lineType=cv2.LINE_AA)
        if point_radius > 0:
            for x, y in ipts:
                cv2.circle(canvas, (int(x), int(y)), point_radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    return canvas


# -------------------- 论文排版 --------------------
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


def make_panel(img, title, panel_size=(420, 420), title_h=52):
    font = _find_font(24)
    canvas = Image.new("RGB", (panel_size[0], panel_size[1] + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_size[0], title_h), fill=(245, 245, 245))
    draw.text((12, 11), title, fill=(30, 30, 30), font=font)

    im = to_rgb_pil(img)
    im.thumbnail(panel_size, Image.Resampling.LANCZOS)
    x = (panel_size[0] - im.width) // 2
    y = title_h + (panel_size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def make_compare_figure(left_img, mid_img, right_img, caption="图4-4 路径提取与矢量化过程示意图"):
    left = make_panel(left_img, "骨架/轮廓提取结果")
    mid = make_panel(mid_img, "原始路径点/轮廓线")
    right = make_panel(right_img, "Douglas-Peucker简化结果")

    gap = 20
    caption_h = 56
    w = left.width + mid.width + right.width + gap * 2
    h = max(left.height, mid.height, right.height) + caption_h
    fig = Image.new("RGB", (w, h), (255, 255, 255))
    fig.paste(left, (0, 0))
    fig.paste(mid, (left.width + gap, 0))
    fig.paste(right, (left.width + gap + mid.width + gap, 0))

    draw = ImageDraw.Draw(fig)
    font = _find_font(24)
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    ty = max(left.height, mid.height, right.height) + 12
    draw.text((tx, ty), caption, fill=(20, 20, 20), font=font)
    return fig


# -------------------- 核心流程 --------------------
def generate_vectorize_figure(
    input_path,
    outdir,
    auto_crop=True,
    exact_code_mode=True,
    route_mode="suzuki",
    contour_source="skeleton",
    retrieval_mode="list",
    epsilon_px=2.0,
    min_points=8,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    image_bgr = imread_unicode(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("图片读取失败")

    binary_img, extras = preprocess_like_main_code(
        image_bgr,
        auto_crop=auto_crop,
        exact_code_mode=exact_code_mode,
    )
    skeleton, skel_method = extract_skeleton(binary_img)

    left_show = to_paper_style(skeleton if contour_source == "skeleton" else binary_img, invert=True)

    if route_mode == "suzuki":
        src = skeleton if contour_source == "skeleton" else binary_img
        raw_paths = find_contours_suzuki(src, retrieval_mode=retrieval_mode, min_points=min_points)
        simp_paths = [simplify_polyline_dp(p, epsilon_px=epsilon_px, closed=True) for p in raw_paths]
        route_desc = f"Suzuki-Abe轮廓提取({retrieval_mode}) + Douglas-Peucker"
    else:
        raw_paths = [np.asarray(p, dtype=np.float32) for p in skeleton_to_paths(skeleton, min_len=max(6, min_points))]
        simp_paths = [simplify_polyline_dp(p, epsilon_px=epsilon_px, closed=False) for p in raw_paths]
        route_desc = "当前代码骨架路径提取 + Douglas-Peucker"

    mid_show = draw_paths_canvas(left_show, raw_paths, line_thickness=1, point_radius=0, connect=True)
    right_show = draw_paths_canvas(left_show, simp_paths, line_thickness=2, point_radius=0, connect=True)

    fig = make_compare_figure(left_show, mid_show, right_show)

    fig_path = outdir / f"{stem}_图4-4_路径提取与矢量化过程示意图.png"
    left_path = outdir / f"{stem}_01_骨架或轮廓输入图.png"
    mid_path = outdir / f"{stem}_02_原始路径点或轮廓线.png"
    right_path = outdir / f"{stem}_03_Douglas-Peucker简化路径.png"

    fig.save(fig_path)
    imwrite_unicode(str(left_path), left_show)
    imwrite_unicode(str(mid_path), mid_show)
    imwrite_unicode(str(right_path), right_show)

    info = {
        "fig_path": str(fig_path),
        "left_path": str(left_path),
        "mid_path": str(mid_path),
        "right_path": str(right_path),
        "crop_box": tuple(int(v) for v in extras["crop_box"]),
        "skeleton_method": skel_method,
        "route_desc": route_desc,
        "raw_count": len(raw_paths),
        "simp_count": len(simp_paths),
        "epsilon_px": float(epsilon_px),
        "contour_source": contour_source,
        "route_mode": route_mode,
    }
    return fig, info


# -------------------- GUI --------------------
class Exp44GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("4.4 路径提取与矢量化小程序")
        self.root.geometry("1180x860")

        self.input_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value=str(Path.cwd() / "exp44_output"))
        self.auto_crop_var = tk.BooleanVar(value=True)
        self.exact_code_var = tk.BooleanVar(value=True)
        self.route_mode_var = tk.StringVar(value="suzuki")
        self.contour_source_var = tk.StringVar(value="skeleton")
        self.retrieval_mode_var = tk.StringVar(value="list")
        self.epsilon_var = tk.DoubleVar(value=2.0)
        self.min_points_var = tk.IntVar(value=8)

        self.preview_label = None
        self._preview_imgtk = None

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="输入图片:").pack(side="left")
        ttk.Entry(row1, textvariable=self.input_var, width=90).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row1, text="选择图片", command=self.choose_input).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="输出目录:").pack(side="left")
        ttk.Entry(row2, textvariable=self.outdir_var, width=88).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row2, text="选择目录", command=self.choose_outdir).pack(side="left")

        opts = ttk.LabelFrame(top, text="参数设置", padding=10)
        opts.pack(fill="x", pady=8)

        row3 = ttk.Frame(opts)
        row3.pack(fill="x", pady=4)
        ttk.Checkbutton(row3, text="自动裁剪文字区域", variable=self.auto_crop_var).pack(side="left", padx=4)
        ttk.Checkbutton(row3, text="完全沿用主程序形态学（闭运算+膨胀）", variable=self.exact_code_var).pack(side="left", padx=12)

        row4 = ttk.Frame(opts)
        row4.pack(fill="x", pady=4)
        ttk.Label(row4, text="路径提取方式:").pack(side="left")
        ttk.Combobox(row4, textvariable=self.route_mode_var, values=["suzuki", "codepath"], width=12, state="readonly").pack(side="left", padx=6)
        ttk.Label(row4, text="输入源:").pack(side="left", padx=(16, 0))
        ttk.Combobox(row4, textvariable=self.contour_source_var, values=["skeleton", "binary"], width=12, state="readonly").pack(side="left", padx=6)
        ttk.Label(row4, text="Suzuki检索模式:").pack(side="left", padx=(16, 0))
        ttk.Combobox(row4, textvariable=self.retrieval_mode_var, values=["list", "external", "tree"], width=10, state="readonly").pack(side="left", padx=6)

        row5 = ttk.Frame(opts)
        row5.pack(fill="x", pady=4)
        ttk.Label(row5, text="Douglas-Peucker epsilon(px):").pack(side="left")
        ttk.Entry(row5, textvariable=self.epsilon_var, width=8).pack(side="left", padx=6)
        ttk.Label(row5, text="最小点数:").pack(side="left", padx=(16, 0))
        ttk.Entry(row5, textvariable=self.min_points_var, width=8).pack(side="left", padx=6)
        ttk.Label(
            row5,
            text="建议：论文图默认用 suzuki + skeleton + epsilon=2.0 即可。",
            foreground="#666666",
        ).pack(side="left", padx=12)

        row6 = ttk.Frame(top)
        row6.pack(fill="x", pady=8)
        ttk.Button(row6, text="生成图4-4", command=self.run).pack(side="left", padx=4)
        ttk.Button(row6, text="打开输出目录", command=self.open_outdir_hint).pack(side="left", padx=4)

        self.info_text = tk.Text(self.root, height=9)
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
        view.thumbnail((1080, 600), Image.Resampling.LANCZOS)
        self._preview_imgtk = ImageTk.PhotoImage(view)
        self.preview_label.configure(image=self._preview_imgtk)

    def run(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("提示", "请先选择输入图片")
            return
        try:
            self.info_text.delete("1.0", "end")
            fig, info = generate_vectorize_figure(
                input_path=input_path,
                outdir=self.outdir_var.get().strip() or "exp44_output",
                auto_crop=self.auto_crop_var.get(),
                exact_code_mode=self.exact_code_var.get(),
                route_mode=self.route_mode_var.get(),
                contour_source=self.contour_source_var.get(),
                retrieval_mode=self.retrieval_mode_var.get(),
                epsilon_px=float(self.epsilon_var.get()),
                min_points=int(self.min_points_var.get()),
            )
            self.show_preview(fig)
            self.log(f"已生成总图: {info['fig_path']}")
            self.log(f"裁剪区域: {info['crop_box']}")
            self.log(f"骨架方法: {info['skeleton_method']}")
            self.log(f"路径方式: {info['route_desc']}")
            self.log(f"原始路径/轮廓数量: {info['raw_count']}")
            self.log(f"简化后路径数量: {info['simp_count']}")
            self.log(f"epsilon(px): {info['epsilon_px']}")
            messagebox.showinfo("完成", f"图4-4已生成：\n{info['fig_path']}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    Exp44GUI().mainloop()
