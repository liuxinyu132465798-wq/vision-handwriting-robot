# -*- coding: utf-8 -*-
import math
import os
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


def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


# ---------- 预处理与骨架 ----------
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
        "close_only": close_only,
    }
    return morph, extras


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
                        if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
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


# ---------- 路径提取 / 优化 / 映射 ----------
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


def path_length(path):
    if len(path) < 2:
        return 0.0
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


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
                best_dist, best_i, best_reverse = d0, i, False
            if d1 < best_dist:
                best_dist, best_i, best_reverse = d1, i, True
        best = remaining.pop(best_i)
        if best_reverse:
            best = best[::-1]
        out.append(best)
        current_end = best[-1]
    return [p.tolist() for p in out]


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _vec_norm(v):
    return math.hypot(v[0], v[1])


def _angle_deg(v1, v2):
    n1 = _vec_norm(v1)
    n2 = _vec_norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(c))


def merge_close_strokes(strokes_mm, strict_gap=0.45, relaxed_gap=0.90, min_stroke_mm=0.08):
    if not strokes_mm:
        return strokes_mm
    short_len = max(0.18, min_stroke_mm * 3.0)
    angle_limit = 70.0
    merged = [list(strokes_mm[0])]
    for nxt in strokes_mm[1:]:
        cur = merged[-1]
        gap_start = math.dist(cur[-1], nxt[0])
        gap_end = math.dist(cur[-1], nxt[-1])
        if gap_end < gap_start:
            nxt = list(reversed(nxt))
            gap = gap_end
        else:
            gap = gap_start
        cur_len = path_length(cur)
        nxt_len = path_length(nxt)
        should_merge = False
        if gap <= strict_gap:
            should_merge = True
        elif gap <= relaxed_gap:
            bridge = _vec(cur[-1], nxt[0])
            cur_dir = _vec(cur[-2], cur[-1]) if len(cur) >= 2 else bridge
            nxt_dir = _vec(nxt[0], nxt[1]) if len(nxt) >= 2 else bridge
            a1 = _angle_deg(cur_dir, bridge)
            a2 = _angle_deg(bridge, nxt_dir)
            if (a1 <= angle_limit and a2 <= angle_limit) or min(cur_len, nxt_len) <= short_len:
                should_merge = True
        if should_merge:
            if gap > 1e-9 and cur[-1] != nxt[0]:
                cur.append(nxt[0])
            cur.extend(nxt[1:])
        else:
            merged.append(list(nxt))
    return merged


def optimize_strokes_for_output(strokes_mm, resample_mm=0.28, min_stroke_mm=0.08, merge_gap_mm=0.45, merge_gap_relaxed_mm=0.90):
    if not strokes_mm:
        return []
    step = max(0.12, float(resample_mm))
    min_stroke = max(0.05, float(min_stroke_mm))
    out = []
    for s in strokes_mm:
        if len(s) < 2:
            continue
        ss = smooth_path(s)
        ss = resample_path(ss, step=step)
        ss = clean_path(ss, min_dist=max(0.05, step * 0.55))
        if len(ss) >= 2 and path_length(ss) >= min_stroke:
            out.append(ss)
    if not out:
        return []
    out = sort_paths(out)
    out = merge_close_strokes(out, strict_gap=merge_gap_mm, relaxed_gap=merge_gap_relaxed_mm, min_stroke_mm=min_stroke_mm)
    out = [clean_path(s, min_dist=max(0.05, step * 0.55)) for s in out]
    out = [s for s in out if len(s) >= 2 and path_length(s) >= min_stroke]
    out = sort_paths(out)
    out = merge_close_strokes(out, strict_gap=merge_gap_mm, relaxed_gap=merge_gap_relaxed_mm, min_stroke_mm=min_stroke_mm)
    out = [clean_path(s, min_dist=max(0.05, step * 0.55)) for s in out]
    out = [s for s in out if len(s) >= 2 and path_length(s) >= min_stroke]
    return out


def normalize_strokes_to_mm(strokes, target_box=13.0, resample_mm=0.28, min_stroke_mm=0.08, merge_gap_mm=0.45, merge_gap_relaxed_mm=0.90):
    xs = [x for s in strokes for x, _ in s]
    ys = [y for s in strokes for _, y in s]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    scale = target_box / max(width, height)
    out = []
    for s in strokes:
        ss = [((x - min_x) * scale, (y - min_y) * scale) for x, y in s]
        if len(ss) >= 2:
            out.append(ss)
    out = optimize_strokes_for_output(out, resample_mm, min_stroke_mm, merge_gap_mm, merge_gap_relaxed_mm)
    if not out:
        return [], 0.0, 0.0, scale, (min_x, min_y)
    xs2 = [x for s in out for x, _ in s]
    ys2 = [y for s in out for _, y in s]
    w_mm = max(xs2) - min(xs2)
    h_mm = max(ys2) - min(ys2)
    return out, w_mm, h_mm, scale, (min_x, min_y)


def write_gcode_from_strokes_mm(strokes_mm, out_path, feed=500, pen_down_cmd="M3 S90", pen_up_cmd="M5"):
    if not strokes_mm:
        raise ValueError("没有可输出的笔画")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"G1 F{int(round(feed))}\n")
        f.write("G92 X0 Y0\n")
        f.write("G21\n")
        f.write("G90\n")
        f.write("M5\n")
        for s in strokes_mm:
            if len(s) < 2:
                continue
            x0, y0 = s[0]
            f.write(f"G0 X{x0:.2f} Y{-y0:.2f}\n")
            f.write(pen_down_cmd + "\n")
            for x, y in s[1:]:
                f.write(f"G1 X{x:.2f} Y{-y:.2f}\n")
            f.write(pen_up_cmd + "\n")
        f.write("G0 X0 Y0\n")


# ---------- 绘图 ----------
def _find_font(size=24, mono=False):
    candidates = [
        r"C:\\Windows\\Fonts\\consola.ttf" if mono else r"C:\\Windows\\Fonts\\msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if mono else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_strokes(draw, box, strokes, mode="px"):
    x0, y0, x1, y1 = box
    pad = 26
    ix0, iy0, ix1, iy1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
    draw.rectangle(box, outline=(210, 210, 210), width=2)
    xs = [x for s in strokes for x, _ in s]
    ys = [y for s in strokes for _, y in s]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((ix1 - ix0) / span_x, (iy1 - iy0) / span_y)
    axis_font = _find_font(18)
    axis_color = (120, 120, 120)
    if mode == "px":
        ox, oy = ix0, iy0
        draw.line([(ox, oy), (ox + 60, oy)], fill=axis_color, width=2)
        draw.line([(ox, oy), (ox, oy + 60)], fill=axis_color, width=2)
        draw.text((ox + 64, oy - 10), "x", fill=axis_color, font=axis_font)
        draw.text((ox - 12, oy + 64), "y", fill=axis_color, font=axis_font)
    else:
        ox, oy = ix0, iy1
        draw.line([(ox, oy), (ox + 60, oy)], fill=axis_color, width=2)
        draw.line([(ox, oy), (ox, oy - 60)], fill=axis_color, width=2)
        draw.text((ox + 64, oy - 10), "X", fill=axis_color, font=axis_font)
        draw.text((ox - 12, oy - 82), "Y", fill=axis_color, font=axis_font)
    for s in strokes:
        pts = []
        for x, y in s:
            if mode == "px":
                px = ix0 + (x - min_x) * scale
                py = iy0 + (y - min_y) * scale
            else:
                px = ix0 + (x - min_x) * scale
                py = iy1 - (y - min_y) * scale
            pts.append((float(px), float(py)))
        if len(pts) >= 2:
            draw.line(pts, fill=(0, 0, 0), width=3, joint="curve")


def make_mapping_panel(pixel_strokes, strokes_mm, w_mm, h_mm, scale_px_to_mm, origin_px):
    W, H = 1080, 470
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _find_font(28)
    text_font = _find_font(20)
    small_font = _find_font(18)
    draw.rectangle((0, 0, W, 56), fill=(245, 245, 245))
    draw.text((16, 14), "坐标映射示意", fill=(20, 20, 20), font=title_font)
    left_box = (28, 88, 430, 388)
    right_box = (650, 88, 1052, 388)
    draw.text((left_box[0], 62), "图像坐标系（像素）", fill=(40, 40, 40), font=text_font)
    draw.text((right_box[0], 62), "机械工作平面坐标系（毫米）", fill=(40, 40, 40), font=text_font)
    render_strokes(draw, left_box, pixel_strokes, mode="px")
    render_strokes(draw, right_box, strokes_mm, mode="mm")
    arrow_y = 220
    draw.line([(455, arrow_y), (620, arrow_y)], fill=(70, 70, 70), width=4)
    draw.polygon([(620, arrow_y), (600, arrow_y - 10), (600, arrow_y + 10)], fill=(70, 70, 70))
    draw.text((470, 120), "归一化 + 缩放", fill=(60, 60, 60), font=text_font)
    draw.text((446, 160), "x_mm = (x - min_x) × scale", fill=(60, 60, 60), font=small_font)
    draw.text((446, 190), "y_mm = (y - min_y) × scale", fill=(60, 60, 60), font=small_font)
    draw.text((462, 245), f"scale ≈ {scale_px_to_mm:.4f} mm/px", fill=(60, 60, 60), font=small_font)
    draw.text((456, 275), "G-code 输出时使用 Y = -y_mm", fill=(120, 0, 0), font=small_font)
    draw.text((34, 404), f"原点：裁剪区左上角  ({origin_px[0]:.1f}, {origin_px[1]:.1f}) px", fill=(80, 80, 80), font=small_font)
    draw.text((650, 404), f"映射后包围盒：W ≈ {w_mm:.2f} mm, H ≈ {h_mm:.2f} mm", fill=(80, 80, 80), font=small_font)
    return img


def make_gcode_panel(gcode_lines, out_name="handwriting.gcode"):
    W, H = 1080, 360
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _find_font(28)
    mono_font = _find_font(21, mono=True)
    small_font = _find_font(18)
    draw.rectangle((0, 0, W, 56), fill=(245, 245, 245))
    draw.text((16, 14), f"生成的 {out_name} 典型指令", fill=(20, 20, 20), font=title_font)
    code_box = (30, 78, 1050, 326)
    draw.rounded_rectangle(code_box, radius=12, fill=(25, 28, 34), outline=(60, 65, 72), width=2)
    x0, y0 = code_box[0] + 18, code_box[1] + 16
    for idx, line in enumerate(gcode_lines[:10], start=1):
        draw.text((x0, y0 + (idx - 1) * 24), f"{idx:>2}  {line}", fill=(230, 232, 235), font=mono_font)
    draw.text((30, 332), "包含快速定位 G0、书写运动 G1、落笔指令 M3 S90 与抬笔指令 M5。", fill=(80, 80, 80), font=small_font)
    return img


def make_full_figure(mapping_img, gcode_img, caption="图4-6 路径坐标映射与 G-code 生成示意图"):
    gap, caption_h = 18, 60
    W = max(mapping_img.width, gcode_img.width)
    H = mapping_img.height + gap + gcode_img.height + caption_h
    fig = Image.new("RGB", (W, H), (255, 255, 255))
    fig.paste(mapping_img, (0, 0))
    fig.paste(gcode_img, (0, mapping_img.height + gap))
    draw = ImageDraw.Draw(fig)
    font = _find_font(24)
    bbox = draw.textbbox((0, 0), caption, font=font)
    draw.text(((W - (bbox[2] - bbox[0])) // 2, mapping_img.height + gap + gcode_img.height + 14), caption, fill=(20, 20, 20), font=font)
    return fig


# ---------- 核心流程 ----------
def generate_gcode_demo_figure(input_path, outdir, auto_crop=True, exact_code_mode=True, target_box_mm=13.0, feed=500, pen_down_cmd="M3 S90", pen_up_cmd="M5"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    image_bgr = imread_unicode(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("图片读取失败")

    binary_roi, extras = preprocess_like_main_code(image_bgr, auto_crop=auto_crop, exact_code_mode=exact_code_mode)
    skel, skel_method = extract_skeleton(binary_roi)
    raw_paths = [p for p in skeleton_to_paths(skel, min_len=6) if len(p) >= 2]
    if not raw_paths:
        raise ValueError("没有提取到有效路径")

    strokes_mm, w_mm, h_mm, scale_px_to_mm, origin_px = normalize_strokes_to_mm(raw_paths, target_box=target_box_mm)
    if not strokes_mm:
        raise ValueError("路径映射到毫米坐标后为空")

    gcode_path = outdir / "handwriting.gcode"
    write_gcode_from_strokes_mm(strokes_mm, gcode_path, feed=feed, pen_down_cmd=pen_down_cmd, pen_up_cmd=pen_up_cmd)
    gcode_lines = [line.strip() for line in gcode_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    mapping_img = make_mapping_panel(raw_paths, strokes_mm, w_mm, h_mm, scale_px_to_mm, origin_px)
    gcode_img = make_gcode_panel(gcode_lines, out_name=gcode_path.name)
    full_fig = make_full_figure(mapping_img, gcode_img)

    mapping_path = outdir / f"{stem}_01_坐标映射示意图.png"
    gcode_shot_path = outdir / f"{stem}_02_G-code指令截图.png"
    fig_path = outdir / f"{stem}_图4-6_路径坐标映射与_G-code_生成示意图.png"
    mapping_img.save(mapping_path)
    gcode_img.save(gcode_shot_path)
    full_fig.save(fig_path)

    info = {
        "mapping_path": str(mapping_path),
        "gcode_shot_path": str(gcode_shot_path),
        "gcode_path": str(gcode_path),
        "fig_path": str(fig_path),
        "crop_box": extras["crop_box"],
        "skeleton_method": skel_method,
        "raw_path_count": len(raw_paths),
        "stroke_count_mm": len(strokes_mm),
        "target_box_mm": float(target_box_mm),
        "mapped_width_mm": float(w_mm),
        "mapped_height_mm": float(h_mm),
        "feed": int(feed),
        "pen_down_cmd": pen_down_cmd,
        "pen_up_cmd": pen_up_cmd,
        "scale_px_to_mm": float(scale_px_to_mm),
    }
    return full_fig, info


# ---------- GUI ----------
class Exp46GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("4.6 路径坐标映射与 G-code 生成示意图")
        self.root.geometry("1240x920")
        self.input_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value=str(Path.cwd() / "exp46_output"))
        self.auto_crop_var = tk.BooleanVar(value=True)
        self.exact_code_var = tk.BooleanVar(value=True)
        self.target_box_var = tk.DoubleVar(value=13.0)
        self.feed_var = tk.IntVar(value=500)
        self.pen_down_var = tk.StringVar(value="M3 S90")
        self.pen_up_var = tk.StringVar(value="M5")
        self._preview_imgtk = None
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        row1 = ttk.Frame(top); row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="输入图片:").pack(side="left")
        ttk.Entry(row1, textvariable=self.input_var, width=92).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row1, text="选择图片", command=self.choose_input).pack(side="left")
        row2 = ttk.Frame(top); row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="输出目录:").pack(side="left")
        ttk.Entry(row2, textvariable=self.outdir_var, width=90).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row2, text="选择目录", command=self.choose_outdir).pack(side="left")
        opts = ttk.LabelFrame(top, text="参数设置", padding=10); opts.pack(fill="x", pady=8)
        row3 = ttk.Frame(opts); row3.pack(fill="x", pady=4)
        ttk.Checkbutton(row3, text="自动裁剪文字区域", variable=self.auto_crop_var).pack(side="left", padx=4)
        ttk.Checkbutton(row3, text="完全沿用主程序形态学（闭运算+膨胀）", variable=self.exact_code_var).pack(side="left", padx=12)
        row4 = ttk.Frame(opts); row4.pack(fill="x", pady=4)
        ttk.Label(row4, text="目标字框(mm):").pack(side="left")
        ttk.Entry(row4, textvariable=self.target_box_var, width=10).pack(side="left", padx=6)
        ttk.Label(row4, text="进给速度 F:").pack(side="left", padx=(16, 0))
        ttk.Entry(row4, textvariable=self.feed_var, width=10).pack(side="left", padx=6)
        ttk.Label(row4, text="落笔指令:").pack(side="left", padx=(16, 0))
        ttk.Entry(row4, textvariable=self.pen_down_var, width=12).pack(side="left", padx=6)
        ttk.Label(row4, text="抬笔指令:").pack(side="left", padx=(16, 0))
        ttk.Entry(row4, textvariable=self.pen_up_var, width=10).pack(side="left", padx=6)
        row5 = ttk.Frame(top); row5.pack(fill="x", pady=8)
        ttk.Button(row5, text="生成图4-6", command=self.run).pack(side="left", padx=4)
        self.info_text = tk.Text(self.root, height=10); self.info_text.pack(fill="x", padx=10, pady=6)
        preview_frame = ttk.LabelFrame(self.root, text="预览", padding=10); preview_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.preview_label = ttk.Label(preview_frame); self.preview_label.pack(expand=True)

    def choose_input(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if path: self.input_var.set(path)

    def choose_outdir(self):
        path = filedialog.askdirectory()
        if path: self.outdir_var.set(path)

    def show_preview(self, pil_img):
        view = pil_img.copy(); view.thumbnail((1120, 700), Image.Resampling.LANCZOS)
        self._preview_imgtk = ImageTk.PhotoImage(view)
        self.preview_label.configure(image=self._preview_imgtk)

    def log(self, msg):
        self.info_text.insert("end", msg + "\n")
        self.info_text.see("end")
        self.root.update_idletasks()

    def run(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("提示", "请先选择输入图片"); return
        try:
            self.info_text.delete("1.0", "end")
            fig, info = generate_gcode_demo_figure(
                input_path=input_path,
                outdir=self.outdir_var.get().strip() or "exp46_output",
                auto_crop=self.auto_crop_var.get(),
                exact_code_mode=self.exact_code_var.get(),
                target_box_mm=float(self.target_box_var.get()),
                feed=int(self.feed_var.get()),
                pen_down_cmd=self.pen_down_var.get().strip() or "M3 S90",
                pen_up_cmd=self.pen_up_var.get().strip() or "M5",
            )
            self.show_preview(fig)
            self.log(f"已生成总图: {info['fig_path']}")
            self.log(f"G-code 文件: {info['gcode_path']}")
            self.log(f"裁剪区域: {info['crop_box']}")
            self.log(f"骨架方法: {info['skeleton_method']}")
            self.log(f"像素路径数: {info['raw_path_count']}")
            self.log(f"毫米笔画数: {info['stroke_count_mm']}")
            self.log(f"映射尺寸: {info['mapped_width_mm']:.2f} mm × {info['mapped_height_mm']:.2f} mm")
            self.log(f"目标字框: {info['target_box_mm']:.2f} mm")
            self.log(f"进给速度 F: {info['feed']}")
            self.log(f"落笔/抬笔: {info['pen_down_cmd']} / {info['pen_up_cmd']}")
            messagebox.showinfo("完成", f"图4-6已生成：\n{info['fig_path']}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    Exp46GUI().mainloop()
