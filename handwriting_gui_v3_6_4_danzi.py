import json
import math
import os
import re
import shlex
import subprocess
import threading
import shutil
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

try:
    from scipy.signal import savgol_filter
except ImportError:
    print("请先运行：pip install scipy")
    savgol_filter = None

Point = Tuple[float, float]
Stroke = List[Point]


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class HandwritingGCodeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("手写字转 G-code 上位机 v3.6.4（Bachin固定13mm + 少抬笔 + .nc输出版）")
        self.root.geometry("1280x900")

        self.scale_var = tk.DoubleVar(value=0.25)
        self.max_width_var = tk.DoubleVar(value=180)
        self.target_box_mm_var = tk.DoubleVar(value=13.0)
        self.resample_mm_var = tk.DoubleVar(value=0.28)
        self.min_stroke_mm_var = tk.DoubleVar(value=0.08)
        self.merge_gap_mm_var = tk.DoubleVar(value=0.45)
        self.merge_gap_relaxed_mm_var = tk.DoubleVar(value=0.90)
        self.pen_down_var = tk.StringVar(value="M3 S90")
        self.pen_up_var = tk.StringVar(value="M5")
        self.feedrate_var = tk.DoubleVar(value=500)
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Desktop" / "Gcode_Output"))
        self.camera_index = tk.IntVar(value=0)

        self.inksight_json_var = tk.StringVar(value="")
        self.inksight_mode_var = tk.StringVar(value="recognize")
        self.inksight_use_roi_var = tk.BooleanVar(value=True)
        self.inksight_distro_var = tk.StringVar(value="Ubuntu-22.04")
        self.inksight_venv_var = tk.StringVar(value="~/inksight-env-2204")
        self.inksight_runner_var = tk.StringVar(value="/mnt/c/Users/刘欣毓/Desktop/local_inksight_runner.py")
        self.inksight_segmentor_var = tk.StringVar(value="tesseract")
        self.inksight_lang_var = tk.StringVar(value="chi_sim+eng")

        self.cap = None
        self.preview_running = False
        self.current_photo = None
        self.current_path = None

        self.polys: List[Stroke] = []
        self.crop_offset = (0, 0)
        self.last_roi = None
        self.last_binary = None
        self.last_skeleton = None

        self.inksight_strokes_raw: List[Stroke] = []
        self.inksight_strokes_pixels: List[Stroke] = []
        self.inksight_coord_mode = "unknown"
        self.inksight_overlay_offset = (0, 0)
        self.inksight_input_shape = None
        self.inksight_last_jsons = {}
        self.inksight_last_mode = None
        self.inksight_last_preview = None

        self.setup_ui()

    # ---------- UI ----------
    def setup_ui(self):
        main_scroll = ScrollableFrame(self.root)
        main_scroll.pack(fill="both", expand=True)

        param_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="参数设置")
        param_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(param_frame, text="缩放(mm/pixel):").grid(row=0, column=0, padx=5)
        ttk.Entry(param_frame, textvariable=self.scale_var, width=10).grid(row=0, column=1)
        ttk.Label(param_frame, text="最大宽度(mm):").grid(row=0, column=2, padx=5)
        ttk.Entry(param_frame, textvariable=self.max_width_var, width=10).grid(row=0, column=3)
        ttk.Label(param_frame, text="G1速度:").grid(row=0, column=4, padx=5)
        ttk.Entry(param_frame, textvariable=self.feedrate_var, width=10).grid(row=0, column=5)

        ttk.Label(param_frame, text="落笔指令:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.pen_down_var, width=15).grid(row=1, column=1)
        ttk.Label(param_frame, text="抬笔指令:").grid(row=1, column=2, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.pen_up_var, width=15).grid(row=1, column=3)
        ttk.Label(param_frame, text="建议：Bachin/你的机器先用 M3 S90 / M5").grid(row=1, column=4, columnspan=2, sticky="w")

        ttk.Label(param_frame, text="固定外接框(mm):").grid(row=2, column=0, padx=5, pady=5)
        ttk.Label(param_frame, text="13 × 13（锁定）").grid(row=2, column=1, sticky="w")
        ttk.Label(param_frame, text="重采样(mm):").grid(row=2, column=2, padx=5)
        ttk.Entry(param_frame, textvariable=self.resample_mm_var, width=10).grid(row=2, column=3)
        ttk.Label(param_frame, text="最短笔画(mm):").grid(row=2, column=4, padx=5)
        ttk.Entry(param_frame, textvariable=self.min_stroke_mm_var, width=10).grid(row=2, column=5)

        ttk.Label(param_frame, text="不断笔连接(mm):").grid(row=3, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.merge_gap_mm_var, width=10).grid(row=3, column=1)
        ttk.Label(param_frame, text="宽松连接(mm):").grid(row=3, column=2, padx=5)
        ttk.Entry(param_frame, textvariable=self.merge_gap_relaxed_mm_var, width=10).grid(row=3, column=3)
        ttk.Label(param_frame, text="说明：已固定输出到约 13×13 mm；这两个值越大，抬笔越少，但更容易误连笔画。").grid(row=3, column=4, columnspan=2, sticky="w")

        output_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="输出文件夹设置")
        output_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(output_frame, text="当前文件夹:").pack(side="left")
        self.output_label = ttk.Label(output_frame, text=self.output_dir_var.get())
        self.output_label.pack(side="left", padx=10)
        ttk.Button(output_frame, text="选择文件夹", command=self.choose_output_dir).pack(side="left")

        img_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="1. 本地图片模式（传统骨架路线）")
        img_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(img_frame, text="📂 选择图片", command=self.select_image).pack(pady=8)
        self.img_label = ttk.Label(img_frame, text="未选择图片")
        self.img_label.pack()
        btn_frame = ttk.Frame(img_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="👁️ 当前图片预览", command=self.show_preview).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🧪 查看处理中间结果", command=self.show_debug_preview).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🚀 传统算法生成 G-code", command=self.process_current_image).pack(side="left", padx=5)

        cam_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="2. 摄像头模式")
        cam_frame.pack(fill="x", padx=10, pady=5)
        cam_sub = ttk.Frame(cam_frame)
        cam_sub.pack()
        ttk.Label(cam_sub, text="摄像头索引(0/1/2):").pack(side="left")
        ttk.Entry(cam_sub, textvariable=self.camera_index, width=8).pack(side="left", padx=5)
        ttk.Button(cam_sub, text="打开预览", command=self.start_camera).pack(side="left", padx=10)
        ttk.Button(cam_sub, text="📸 拍照裁剪入本地模式", command=self.capture_to_local_image).pack(side="left")
        self.canvas = tk.Canvas(cam_frame, width=640, height=480, bg="black")
        self.canvas.pack(pady=10)

        gcode_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="3. 独立 G-code 预览模块")
        gcode_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(gcode_frame, text="📂 选择 G-code 文件预览", command=self.preview_any_gcode).pack(pady=10)

        ink_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="4. InkSight 一键整合")
        ink_frame.pack(fill="x", padx=10, pady=5)

        row0 = ttk.Frame(ink_frame)
        row0.pack(fill="x", pady=4)
        ttk.Label(row0, text="模式:").pack(side="left")
        ttk.Combobox(
            row0,
            textvariable=self.inksight_mode_var,
            values=["recognize", "vanilla"],
            width=12,
            state="readonly",
        ).pack(side="left", padx=6)
        ttk.Checkbutton(row0, text="先裁剪文字 ROI 再送 InkSight", variable=self.inksight_use_roi_var).pack(side="left", padx=8)
        ttk.Label(row0, text="分词器:").pack(side="left", padx=(12, 0))
        ttk.Combobox(
            row0,
            textvariable=self.inksight_segmentor_var,
            values=["tesseract", "doctr"],
            width=10,
            state="readonly",
        ).pack(side="left", padx=6)

        row1 = ttk.Frame(ink_frame)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="WSL发行版:").pack(side="left")
        ttk.Entry(row1, textvariable=self.inksight_distro_var, width=16).pack(side="left", padx=6)
        ttk.Label(row1, text="虚拟环境:").pack(side="left")
        ttk.Entry(row1, textvariable=self.inksight_venv_var, width=26).pack(side="left", padx=6)
        ttk.Label(row1, text="语言:").pack(side="left")
        ttk.Entry(row1, textvariable=self.inksight_lang_var, width=16).pack(side="left", padx=6)

        row2 = ttk.Frame(ink_frame)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Runner路径:").pack(side="left")
        ttk.Entry(row2, textvariable=self.inksight_runner_var, width=90).pack(side="left", padx=6, fill="x", expand=True)

        row3 = ttk.Frame(ink_frame)
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="InkSight JSON:").pack(side="left")
        ttk.Entry(row3, textvariable=self.inksight_json_var, width=75).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row3, text="选择 JSON", command=self.choose_inksight_json).pack(side="left", padx=4)
        ttk.Button(row3, text="导入 JSON", command=self.import_inksight_json).pack(side="left", padx=4)

        row4 = ttk.Frame(ink_frame)
        row4.pack(fill="x", pady=8)
        ttk.Button(row4, text="🧠 InkSight-当前模式一键生成", command=self.run_inksight_current_mode).pack(side="left", padx=5)
        ttk.Button(row4, text="🧠 InkSight-双模式都跑", command=self.run_inksight_dual).pack(side="left", padx=5)
        ttk.Button(row4, text="👁️ 预览 InkSight 结果", command=self.preview_inksight_result).pack(side="left", padx=5)
        ttk.Button(row4, text="🖼️ 源图叠加预览", command=self.preview_inksight_overlay).pack(side="left", padx=5)
        ttk.Button(row4, text="✍️ 从已导入 InkSight 结果生成 G-code", command=self.generate_gcode_from_inksight).pack(side="left", padx=5)

        hint = ttk.Label(
            ink_frame,
            text=(
                "说明：当前模式支持 recognize / vanilla。"
                "建议默认勾选 ROI，再用‘双模式都跑’对比。"
                "不再需要手工 PowerShell 生成 JSON。"
            ),
        )
        hint.pack(anchor="w", padx=4, pady=2)

        log_frame = ttk.LabelFrame(main_scroll.scrollable_frame, text="日志")
        log_frame.pack(fill="x", padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=14)
        self.log_text.pack(fill="both", expand=True)

        self.log("✅ v3.6.2 启动成功：已加入 Bachin 风格的少抬笔输出")
        self.log("提示：中间结果现在支持可缩放、可拉伸窗口；摄像头拍照后会先裁剪文字区域，再交给本地图片模式处理。")
        self.root.mainloop()

    def log(self, msg):
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")
        self.root.update()

    def choose_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_var.set(dir_path)
            self.output_label.config(text=dir_path)
            os.makedirs(dir_path, exist_ok=True)
            self.log(f"输出文件夹已更改为: {dir_path}")

    # ---------- Traditional pipeline ----------
    def preprocess_image(self, img):
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=25, sigmaY=25)
        norm = cv2.divide(img, bg, scale=255)
        norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
        norm = cv2.GaussianBlur(norm, (3, 3), 0)

        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        boxes = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 20 and h >= 8 and w >= 2:
                clean[labels == i] = 255
                boxes.append((x, y, w, h))

        if not boxes:
            raise ValueError("没有检测到有效笔迹，请检查拍照亮度、对焦和纸张背景")

        x0 = max(0, min(b[0] for b in boxes) - 15)
        y0 = max(0, min(b[1] for b in boxes) - 15)
        x1 = min(img.shape[1], max(b[0] + b[2] for b in boxes) + 15)
        y1 = min(img.shape[0], max(b[1] + b[3] for b in boxes) + 15)

        roi = clean[y0:y1, x0:x1].copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        roi = cv2.dilate(roi, kernel, iterations=1)

        self.crop_offset = (x0, y0)
        self.last_binary = clean
        self.last_roi = roi
        return roi, (x0, y0, x1, y1)

    def get_color_roi_for_inksight(self, img_path):
        img_data = np.fromfile(img_path, dtype=np.uint8)
        color = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError("图片读取失败")
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        _, crop_box = self.preprocess_image(gray)
        x0, y0, x1, y1 = crop_box
        roi = color[y0:y1, x0:x1].copy()
        return roi, crop_box

    def skeleton_to_paths(self, skel, min_len=6):
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

    def smooth_path(self, path):
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

    def resample_path(self, path, step=1.2):
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

    def clean_path(self, path, min_dist=0.8):
        if len(path) < 2:
            return []
        cleaned = [path[0]]
        for p in path[1:]:
            if math.dist(p, cleaned[-1]) > min_dist:
                cleaned.append(p)
        return cleaned

    def sort_paths(self, polys):
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

    def generate_gcode(self, img_path):
        try:
            self.log(f"正在读取图片: {img_path}")
            img_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("图片读取失败")

            binary_roi, crop_box = self.preprocess_image(img)
            if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "thinning"):
                raise RuntimeError("缺少 cv2.ximgproc.thinning，请安装 opencv-contrib-python")

            skel = cv2.ximgproc.thinning(binary_roi)
            self.last_skeleton = skel.copy()

            raw_paths = self.skeleton_to_paths(skel, min_len=6)
            polys = []
            for path in raw_paths:
                path = self.smooth_path(path)
                path = self.resample_path(path, step=1.2)
                path = self.clean_path(path)
                if len(path) >= 6:
                    polys.append(path)

            if not polys:
                raise ValueError("没有提取到有效路径")

            polys = self.sort_paths(polys)
            self.polys = polys

            base_name = Path(img_path).stem
            out_path = self.generate_gcode_from_polys(polys, f"{base_name}.nc")

            self.log(f"✅ 裁剪区域: {crop_box}")
            self.log(f"✅ 原始识别路径数: {len(polys)}")
            self.log(f"🎉 传统算法输出成功！→ {out_path}")
            return out_path

        except Exception as e:
            self.log(f"❌ 处理失败: {str(e)}")
            messagebox.showerror("错误", f"处理失败:\n{str(e)}")
            return None

    def capture_to_local_image(self):
        if not self.preview_running or self.current_photo is None:
            messagebox.showwarning("提示", "请先打开摄像头预览！")
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_path = output_dir / f"camera_capture_{timestamp}.jpg"
            crop_path = output_dir / f"camera_capture_{timestamp}_crop.jpg"

            ok, raw_buf = cv2.imencode('.jpg', self.current_photo)
            if not ok:
                raise ValueError("拍照原图编码失败")
            raw_buf.tofile(str(raw_path))

            gray = cv2.cvtColor(self.current_photo, cv2.COLOR_BGR2GRAY)
            _, crop_box = self.preprocess_image(gray)
            x0, y0, x1, y1 = crop_box
            roi_color = self.current_photo[y0:y1, x0:x1].copy()
            ok, crop_buf = cv2.imencode('.jpg', roi_color)
            if not ok:
                raise ValueError("裁剪图片编码失败")
            crop_buf.tofile(str(crop_path))

            self.current_path = str(crop_path)
            self.img_label.config(text=f"已选择: {crop_path.name}")
            self.polys = []
            self.last_skeleton = None
            self.inksight_strokes_raw = []
            self.inksight_strokes_pixels = []
            self.inksight_json_var.set("")

            self.log(f"✅ 摄像头原图已保存: {raw_path}")
            self.log(f"✅ 已自动裁剪文字区域: {crop_box}")
            self.log(f"✅ 裁剪图已写入并自动载入本地图片模式: {crop_path}")
            self.log("下一步：直接在“本地图片模式”里点“查看处理中间结果”或“传统算法生成 G-code”。")

            self.show_image_in_scrollable_viewer(
                "摄像头裁剪结果（已载入本地图片模式）",
                roi_color,
                prefer_nearest=False,
            )
        except Exception as e:
            self.log(f"❌ 拍照裁剪失败: {str(e)}")
            messagebox.showerror("错误", str(e))

    # ---------- Generic preview helpers ----------
    def _fit_polys_to_canvas(self, polys, canvas_w=1400, canvas_h=900, padding=40):
        if not polys:
            return None, None, None, None, None

        xs = [x for path in polys for x, _ in path]
        ys = [y for path in polys for _, y in path]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        scale = min((canvas_w - 2 * padding) / width, (canvas_h - 2 * padding) / height)
        thickness = max(1, int(round(scale * 0.10)))
        canvas = np.ones((canvas_h, canvas_w, 3), np.uint8) * 255

        def map_pt(x, y):
            sx = int((x - min_x) * scale + padding)
            sy = int((max_y - y) * scale + padding)
            return sx, sy

        return canvas, map_pt, scale, thickness, (min_x, min_y, max_x, max_y)

    def _make_labeled_panel(self, img_bgr, title, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
        if img_bgr is None:
            raise ValueError("图像为空")
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
        top_bar_h = 42
        panel = np.zeros((h + top_bar_h, w, 3), dtype=np.uint8)
        panel[:] = bg_color
        panel[top_bar_h:top_bar_h + h, :w] = img_bgr
        cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, lineType=cv2.LINE_AA)
        return panel

    def show_image_in_scrollable_viewer(self, title, image_bgr, prefer_nearest=True):
        if image_bgr is None:
            messagebox.showwarning("提示", "没有可显示的图像")
            return

        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        base_image = Image.fromarray(rgb)

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1280x860")
        win.minsize(720, 480)

        toolbar = ttk.Frame(win)
        toolbar.pack(fill="x", padx=8, pady=6)

        ttk.Label(toolbar, text="缩放:").pack(side="left")
        zoom_var = tk.DoubleVar(value=100.0)
        zoom_label = ttk.Label(toolbar, text="100%")
        zoom_label.pack(side="left", padx=(6, 12))

        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(canvas_frame, bg="black")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        x_scroll.pack(side="bottom", fill="x")
        y_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        state = {"photo": None, "suspend": False}

        def render():
            zoom = max(0.1, min(8.0, float(zoom_var.get()) / 100.0))
            new_w = max(1, int(round(base_image.width * zoom)))
            new_h = max(1, int(round(base_image.height * zoom)))
            resample = Image.NEAREST if prefer_nearest else Image.BILINEAR
            disp = base_image.resize((new_w, new_h), resample)
            photo = ImageTk.PhotoImage(disp)
            state["photo"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.configure(scrollregion=(0, 0, new_w, new_h))
            zoom_label.config(text=f"{zoom * 100:.0f}%")

        def set_zoom(percent):
            state["suspend"] = True
            zoom_var.set(max(10.0, min(800.0, percent)))
            state["suspend"] = False
            render()

        def on_scale(_=None):
            if not state["suspend"]:
                render()

        def fit_to_window(_=None):
            win.update_idletasks()
            avail_w = max(200, canvas.winfo_width() - 4)
            avail_h = max(200, canvas.winfo_height() - 4)
            scale = min(avail_w / max(1, base_image.width), avail_h / max(1, base_image.height))
            set_zoom(min(800.0, max(10.0, scale * 100.0)))

        def zoom_in():
            set_zoom(float(zoom_var.get()) * 1.25)

        def zoom_out():
            set_zoom(float(zoom_var.get()) / 1.25)

        ttk.Button(toolbar, text="适应窗口", command=fit_to_window).pack(side="left", padx=4)
        ttk.Button(toolbar, text="100%", command=lambda: set_zoom(100.0)).pack(side="left", padx=4)
        ttk.Button(toolbar, text="放大", command=zoom_in).pack(side="left", padx=4)
        ttk.Button(toolbar, text="缩小", command=zoom_out).pack(side="left", padx=4)
        ttk.Scale(toolbar, from_=10, to=800, orient="horizontal", variable=zoom_var, command=on_scale).pack(side="left", fill="x", expand=True, padx=10)
        ttk.Label(toolbar, text="提示：窗口可自由拉伸，拖动滚动条查看细节。Ctrl+滚轮可缩放。", foreground="#666666").pack(side="right", padx=6)

        def on_mousewheel(event):
            if event.state & 0x0004:
                if event.delta > 0:
                    zoom_in()
                else:
                    zoom_out()
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Shift-MouseWheel>", on_shift_mousewheel)
        win.bind("<Control-plus>", lambda e: zoom_in())
        win.bind("<Control-equal>", lambda e: zoom_in())
        win.bind("<Control-minus>", lambda e: zoom_out())
        win.bind("f", fit_to_window)

        win.after(120, fit_to_window)

    def preview_any_gcode(self):
        gcode_path = filedialog.askopenfilename(filetypes=[("G-code 文件", "*.gcode")])
        if not gcode_path:
            return

        polys = self.parse_gcode_to_polys(gcode_path)
        polys = [p for p in polys if len(p) >= 2]
        if not polys:
            messagebox.showwarning("提示", "该G-code文件没有有效路径")
            return

        canvas, map_pt, scale, thickness, _ = self._fit_polys_to_canvas(polys)
        for path in polys:
            pts = np.array([map_pt(x, y) for x, y in path], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

        title = f"G-code 预览（scale={scale:.2f}px/unit，line={thickness}px）"
        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.log(f"G-code 预览完成: {os.path.basename(gcode_path)}")

    def parse_gcode_to_polys(self, gcode_path):
        polys = []
        current = []
        with open(gcode_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.strip().upper()
                if not line:
                    continue

                is_g0 = line.startswith('G0') or line.startswith('G00')
                is_g1 = line.startswith('G1') or line.startswith('G01')
                if not (is_g0 or is_g1):
                    continue

                xm = re.search(r'X([+-]?\d+(?:\.\d+)?)', line)
                ym = re.search(r'Y([+-]?\d+(?:\.\d+)?)', line)
                if not (xm and ym):
                    continue

                x = float(xm.group(1))
                y = float(ym.group(1))

                if is_g0:
                    if len(current) >= 2:
                        polys.append(current)
                    current = [(x, y)]
                else:
                    if not current:
                        current = [(x, y)]
                    elif (x, y) != current[-1]:
                        current.append((x, y))

        if len(current) >= 2:
            polys.append(current)
        return polys

    def show_preview(self):
        if not self.current_path or not self.polys:
            messagebox.showwarning("提示", "请先生成一次 G-code")
            return

        img_data = np.fromfile(self.current_path, dtype=np.uint8)
        color = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if color is None:
            messagebox.showerror("错误", "无法读取当前图片")
            return

        overlay = color.copy()
        ox, oy = self.crop_offset
        h, w = color.shape[:2]
        thickness = max(1, int(round(min(h, w) / 600)))

        for path in self.polys:
            pts = np.array([[int(round(x + ox)), int(round(y + oy))] for x, y in path], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(overlay, [pts], False, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

        preview = cv2.addWeighted(color, 0.78, overlay, 0.35, 0)
        max_w, max_h = 1400, 900
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imshow("当前图片笔画预览（细线叠加）", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_debug_preview(self):
        if self.last_roi is None or self.last_skeleton is None:
            messagebox.showwarning("提示", "请先成功生成一次 G-code")
            return

        roi_vis = cv2.cvtColor(self.last_roi, cv2.COLOR_GRAY2BGR)
        skel_vis = cv2.cvtColor(self.last_skeleton, cv2.COLOR_GRAY2BGR)

        roi_panel = self._make_labeled_panel(roi_vis, "ROI(binary)")
        skel_panel = self._make_labeled_panel(skel_vis, "Skeleton")

        gap = 12
        panel_h = max(roi_panel.shape[0], skel_panel.shape[0])
        panel_w = roi_panel.shape[1] + gap + skel_panel.shape[1]
        combo = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        combo[:] = (0, 0, 0)
        combo[:roi_panel.shape[0], :roi_panel.shape[1]] = roi_panel
        combo[:skel_panel.shape[0], roi_panel.shape[1] + gap:roi_panel.shape[1] + gap + skel_panel.shape[1]] = skel_panel

        self.show_image_in_scrollable_viewer("处理中间结果（可缩放/可拉伸）", combo, prefer_nearest=True)

    # ---------- InkSight helpers ----------
    def win_to_wsl_path(self, path: str) -> str:
        path = os.path.abspath(path)
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    def _coerce_float(self, v):
        try:
            return float(v)
        except Exception:
            return None

    def _normalize_point(self, p):
        if isinstance(p, dict):
            x = self._coerce_float(p.get("x"))
            y = self._coerce_float(p.get("y"))
            if x is not None and y is not None:
                return (x, y)
            for xk, yk in (("u", "v"), ("0", "1")):
                x = self._coerce_float(p.get(xk))
                y = self._coerce_float(p.get(yk))
                if x is not None and y is not None:
                    return (x, y)
            return None
        if isinstance(p, Sequence) and not isinstance(p, (str, bytes)) and len(p) >= 2:
            x = self._coerce_float(p[0])
            y = self._coerce_float(p[1])
            if x is not None and y is not None:
                return (x, y)
        return None

    def _normalize_stroke(self, obj):
        if isinstance(obj, dict):
            xs, ys = obj.get("x"), obj.get("y")
            if isinstance(xs, Sequence) and isinstance(ys, Sequence) and len(xs) == len(ys) and len(xs) > 0:
                pts = []
                for x, y in zip(xs, ys):
                    xf = self._coerce_float(x)
                    yf = self._coerce_float(y)
                    if xf is not None and yf is not None:
                        pts.append((xf, yf))
                if len(pts) >= 2:
                    return pts
            for key in ("points", "samples", "vertices", "coords"):
                if key in obj:
                    return self._normalize_stroke(obj[key])

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            pts = []
            for p in obj:
                q = self._normalize_point(p)
                if q is not None:
                    pts.append(q)
            if len(pts) >= 2:
                return pts
        return None

    def _extract_strokes_recursive(self, obj):
        keys = (
            "strokes", "ink", "digital_ink", "paths", "polylines", "traces",
            "result", "results", "prediction", "predictions", "words", "lines",
        )
        one = self._normalize_stroke(obj)
        if one is not None:
            return [one]

        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    sub = self._extract_strokes_recursive(obj[key])
                    if sub:
                        return sub
            out = []
            for v in obj.values():
                sub = self._extract_strokes_recursive(v)
                if sub:
                    out.extend(sub)
            return out

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            out = []
            for item in obj:
                sub = self._extract_strokes_recursive(item)
                if sub:
                    out.extend(sub)
            return out
        return []

    def load_inksight_json(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        strokes = self._extract_strokes_recursive(data)
        strokes = [s for s in strokes if len(s) >= 2]
        if not strokes:
            raise ValueError("未能从 InkSight JSON 中解析出笔画")
        return strokes

    def detect_coordinate_mode(self, strokes, image_shape=None):
        xs = [x for s in strokes for x, _ in s]
        ys = [y for s in strokes for _, y in s]
        max_x, max_y = max(xs), max(ys)
        min_x, min_y = min(xs), min(ys)

        if 0.0 <= min_x and 0.0 <= min_y and max_x <= 1.5 and max_y <= 1.5:
            return "normalized"
        if image_shape is not None:
            h, w = image_shape
            if max_x <= w * 1.2 and max_y <= h * 1.2:
                return "pixels"
        return "arbitrary"

    def convert_to_pixels(self, strokes, coord_mode, image_shape=None):
        if coord_mode == "pixels":
            return strokes
        if image_shape is None:
            if coord_mode == "normalized":
                return [[(x * 1000.0, y * 1000.0) for x, y in s] for s in strokes]
            return strokes
        h, w = image_shape
        if coord_mode == "normalized":
            return [[(x * w, y * h) for x, y in s] for s in strokes]
        return strokes

    def choose_inksight_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")])
        if path:
            self.inksight_json_var.set(path)

    def import_inksight_json(self):
        path = self.inksight_json_var.get().strip()
        if not path:
            messagebox.showwarning("提示", "请先选择 InkSight JSON")
            return

        try:
            raw = self.load_inksight_json(path)
            image_shape = self.inksight_input_shape
            self.inksight_coord_mode = self.detect_coordinate_mode(raw, image_shape)
            self.inksight_strokes_raw = raw
            self.inksight_strokes_pixels = self.convert_to_pixels(raw, self.inksight_coord_mode, image_shape)
            self.log(f"✅ 已导入 InkSight JSON：{os.path.basename(path)}")
            self.log(f"✅ 笔画数：{len(self.inksight_strokes_raw)}，坐标模式：{self.inksight_coord_mode}")
        except Exception as e:
            self.log(f"❌ 导入 InkSight JSON 失败: {e}")
            messagebox.showerror("错误", f"导入失败:\n{e}")

    def save_inksight_input(self, img_path, tag):
        output_dir = Path(self.output_dir_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        base = Path(img_path).stem
        roi_path = output_dir / f"{base}_{tag}_input.jpg"

        if self.inksight_use_roi_var.get():
            roi, crop_box = self.get_color_roi_for_inksight(img_path)
            ok, buf = cv2.imencode(".jpg", roi)
            if not ok:
                raise ValueError("ROI 图片编码失败")
            buf.tofile(str(roi_path))
            self.inksight_overlay_offset = (crop_box[0], crop_box[1])
            self.inksight_input_shape = roi.shape[:2]
            self.log(f"✅ InkSight 输入已裁剪 ROI：{crop_box}")
        else:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("图片读取失败")
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                raise ValueError("原图编码失败")
            buf.tofile(str(roi_path))
            self.inksight_overlay_offset = (0, 0)
            self.inksight_input_shape = img.shape[:2]

        return str(roi_path)

    def get_wsl_venv_python(self, venv_path: str) -> str:
        venv = (venv_path or "").strip().replace('\\', '/')
        if not venv:
            venv = "~/inksight-env-2204"
        venv = venv.rstrip('/')
        if venv.startswith('~/'):
            return f"{venv}/bin/python"
        if venv == '~':
            return "~/bin/python"
        return f"{venv}/bin/python"

    def run_inksight(self, img_path, prompt_mode="recognize", output_suffix=""):
        output_dir = Path(self.output_dir_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        base = Path(img_path).stem
        suffix = output_suffix or prompt_mode
        json_path = str(output_dir / f"{base}_{suffix}.json")

        input_path = self.save_inksight_input(img_path, suffix)
        wsl_img = self.win_to_wsl_path(input_path)
        wsl_json = self.win_to_wsl_path(json_path)
        wsl_runner = self.inksight_runner_var.get().strip()
        distro = self.inksight_distro_var.get().strip() or "Ubuntu-22.04"
        venv = self.inksight_venv_var.get().strip() or "~/inksight-env-2204"
        segmentor = self.inksight_segmentor_var.get().strip() or "tesseract"
        lang = self.inksight_lang_var.get().strip() or "chi_sim+eng"
        python_bin = self.get_wsl_venv_python(venv)

        bash_cmd = (
            f'test -x {python_bin} || {{ echo "WSL虚拟环境Python不存在: {python_bin}"; exit 127; }}; '
            f'{python_bin} {shlex.quote(wsl_runner)} '
            f'--input {shlex.quote(wsl_img)} '
            f'--output {shlex.quote(wsl_json)} '
            f'--mode fullpage '
            f'--segmentor {shlex.quote(segmentor)} '
            f'--prompt-mode {shlex.quote(prompt_mode)} '
            f'--lang {shlex.quote(lang)}'
        )

        self.log(f"🚀 开始运行 InkSight：{prompt_mode}")
        self.log(f"    输入图: {input_path}")
        self.log(f"    WSL发行版: {distro}")
        self.log(f"    WSL虚拟环境: {venv}")
        result = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", bash_cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        if result.stdout:
            for line in result.stdout.splitlines():
                if line.strip():
                    self.log(line)
        if result.returncode != 0:
            if result.stderr:
                self.log(result.stderr)
            raise RuntimeError(result.stderr or result.stdout or f"WSL 命令失败，退出码 {result.returncode}")

        if not Path(json_path).exists():
            raise FileNotFoundError(f"未发现 InkSight JSON 输出: {json_path}")

        self.inksight_last_jsons[prompt_mode] = json_path
        self.inksight_last_mode = prompt_mode
        self.log(f"✅ InkSight 完成: {json_path}")
        return json_path

    def process_with_inksight(self, prompt_mode="recognize"):
        if not self.current_path:
            messagebox.showwarning("提示", "请先选择图片")
            return

        try:
            json_path = self.run_inksight(self.current_path, prompt_mode=prompt_mode)
            self.inksight_json_var.set(json_path)
            self.import_inksight_json()

            base = Path(self.current_path).stem
            self.generate_gcode_from_polys(
                self.inksight_strokes_pixels,
                f"{base}_{prompt_mode}.gcode",
            )
            self.log(f"✅ {prompt_mode} 模式已自动生成 G-code")
        except Exception as e:
            self.log(f"❌ InkSight 处理失败: {e}")
            messagebox.showerror("错误", str(e))

    def run_inksight_current_mode(self):
        threading.Thread(
            target=lambda: self.process_with_inksight(self.inksight_mode_var.get().strip() or "recognize"),
            daemon=True,
        ).start()

    def run_inksight_dual_worker(self):
        if not self.current_path:
            self.root.after(0, lambda: messagebox.showwarning("提示", "请先选择图片"))
            return

        try:
            json_r = self.run_inksight(self.current_path, prompt_mode="recognize")
            json_v = self.run_inksight(self.current_path, prompt_mode="vanilla")
            self.log("✅ 双模式运行完成")
            self.log(f"recognize: {json_r}")
            self.log(f"vanilla:   {json_v}")
            self.root.after(0, lambda: messagebox.showinfo("完成", "双模式 JSON 已生成。\n你可以分别导入对比，或直接查看输出文件夹。"))
        except Exception as e:
            self.log(f"❌ 双模式运行失败: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))

    def run_inksight_dual(self):
        threading.Thread(target=self.run_inksight_dual_worker, daemon=True).start()

    def preview_inksight_result(self):
        if not self.inksight_strokes_pixels:
            messagebox.showwarning("提示", "请先导入 InkSight JSON")
            return

        canvas, map_pt, scale, thickness, _ = self._fit_polys_to_canvas(self.inksight_strokes_pixels)
        for path in self.inksight_strokes_pixels:
            pts = np.array([map_pt(x, y) for x, y in path], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

        title = f"InkSight 笔画预览（scale={scale:.2f}px/unit，line={thickness}px）"
        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def preview_inksight_overlay(self):
        if not self.inksight_strokes_pixels:
            messagebox.showwarning("提示", "请先导入 InkSight JSON")
            return
        if not self.current_path:
            messagebox.showwarning("提示", "请先选择源图片，才能做叠加预览")
            return

        img = cv2.imread(self.current_path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "无法读取源图片")
            return

        overlay = img.copy()
        h, w = img.shape[:2]
        thickness = max(1, int(round(min(h, w) / 600)))
        ox, oy = self.inksight_overlay_offset

        for path in self.inksight_strokes_pixels:
            pts = np.array([[int(round(x + ox)), int(round(y + oy))] for x, y in path], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(overlay, [pts], False, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

        preview = cv2.addWeighted(img, 0.78, overlay, 0.35, 0)
        self.inksight_last_preview = preview.copy()
        max_w, max_h = 1400, 900
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imshow("InkSight 源图叠加预览", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def path_length(self, path):
        if len(path) < 2:
            return 0.0
        return sum(math.dist(a, b) for a, b in zip(path, path[1:]))

    def _vec(self, a, b):
        return (b[0] - a[0], b[1] - a[1])

    def _vec_norm(self, v):
        return math.hypot(v[0], v[1])

    def _angle_deg(self, v1, v2):
        n1 = self._vec_norm(v1)
        n2 = self._vec_norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        return math.degrees(math.acos(c))

    def merge_close_strokes(self, strokes_mm):
        if not strokes_mm:
            return strokes_mm

        strict_gap = max(0.0, float(self.merge_gap_mm_var.get()))
        relaxed_gap = max(strict_gap, float(self.merge_gap_relaxed_mm_var.get()))
        short_len = max(0.18, float(self.min_stroke_mm_var.get()) * 3.0)
        angle_limit = 70.0

        merged = [list(strokes_mm[0])]
        merge_count = 0

        for nxt in strokes_mm[1:]:
            cur = merged[-1]

            gap_start = math.dist(cur[-1], nxt[0])
            gap_end = math.dist(cur[-1], nxt[-1])
            if gap_end < gap_start:
                nxt = list(reversed(nxt))
                gap = gap_end
            else:
                gap = gap_start

            cur_len = self.path_length(cur)
            nxt_len = self.path_length(nxt)

            should_merge = False
            if gap <= strict_gap:
                should_merge = True
            elif gap <= relaxed_gap:
                bridge = self._vec(cur[-1], nxt[0])
                cur_dir = self._vec(cur[-2], cur[-1]) if len(cur) >= 2 else bridge
                nxt_dir = self._vec(nxt[0], nxt[1]) if len(nxt) >= 2 else bridge
                a1 = self._angle_deg(cur_dir, bridge)
                a2 = self._angle_deg(bridge, nxt_dir)
                if (a1 <= angle_limit and a2 <= angle_limit) or min(cur_len, nxt_len) <= short_len:
                    should_merge = True

            if should_merge:
                if gap > 1e-9 and cur[-1] != nxt[0]:
                    cur.append(nxt[0])
                cur.extend(nxt[1:])
                merge_count += 1
            else:
                merged.append(list(nxt))

        self.log(f"✅ 不断笔连接后：{len(strokes_mm)} → {len(merged)} 条笔画（减少 {merge_count} 次抬笔）")
        return merged

    def optimize_strokes_for_output(self, strokes_mm):
        if not strokes_mm:
            return []

        step = max(0.12, float(self.resample_mm_var.get()))
        min_stroke = max(0.05, float(self.min_stroke_mm_var.get()))

        out = []
        for s in strokes_mm:
            if len(s) < 2:
                continue
            ss = self.smooth_path(s)
            ss = self.resample_path(ss, step=step)
            ss = self.clean_path(ss, min_dist=max(0.05, step * 0.55))
            if len(ss) >= 2 and self.path_length(ss) >= min_stroke:
                out.append(ss)

        if not out:
            return []

        out = self.sort_paths(out)
        out = self.merge_close_strokes(out)
        out = [self.clean_path(s, min_dist=max(0.05, step * 0.55)) for s in out]
        out = [s for s in out if len(s) >= 2 and self.path_length(s) >= min_stroke]
        out = self.sort_paths(out)
        out = self.merge_close_strokes(out)
        out = [self.clean_path(s, min_dist=max(0.05, step * 0.55)) for s in out]
        out = [s for s in out if len(s) >= 2 and self.path_length(s) >= min_stroke]
        return out

    def normalize_strokes_to_mm(self, strokes):
        xs = [x for s in strokes for x, _ in s]
        ys = [y for s in strokes for _, y in s]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)

        target_box = 13.0
        scale = target_box / max(width, height)

        out = []
        for s in strokes:
            ss = [((x - min_x) * scale, (y - min_y) * scale) for x, y in s]
            if len(ss) >= 2:
                out.append(ss)

        out = self.optimize_strokes_for_output(out)

        if not out:
            return [], 0.0, 0.0

        xs2 = [x for s in out for x, _ in s]
        ys2 = [y for s in out for _, y in s]
        w_mm = max(xs2) - min(xs2)
        h_mm = max(ys2) - min(ys2)
        return out, w_mm, h_mm

    def write_gcode_from_strokes_mm(self, strokes_mm, out_path):
        if not strokes_mm:
            raise ValueError("没有可输出的笔画")

        feed = int(round(float(self.feedrate_var.get())))

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"G1 F{feed}\n")
            f.write("G92 X0 Y0\n")
            f.write("G21\n")
            f.write("G90\n")
            f.write("M5\n")

            for s in strokes_mm:
                if len(s) < 2:
                    continue
                x0, y0 = s[0]
                f.write(f"G0 X{x0:.2f} Y{-y0:.2f}\n")
                f.write(self.pen_down_var.get() + "\n")
                for x, y in s[1:]:
                    f.write(f"G1 X{x:.2f} Y{-y:.2f}\n")
                f.write(self.pen_up_var.get() + "\n")

            f.write("G0 X0 Y0\n")

    def save_canonical_inksight_json(self, strokes_mm, out_path):
        payload = {
            "format": "canonical_digital_ink_v1",
            "units": "mm",
            "strokes": [
                {"points": [{"x": round(x, 6), "y": round(y, 6)} for x, y in s]}
                for s in strokes_mm
            ],
        }
        Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate_gcode_from_polys(self, polys, output_name="inksight_output.nc"):
        strokes_mm, w_mm, h_mm = self.normalize_strokes_to_mm(polys)
        if not strokes_mm:
            raise ValueError("规范化后没有可输出的笔画")

        output_dir = Path(self.output_dir_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / output_name
        if out_path.suffix.lower() not in (".nc", ".gcode"):
            out_path = out_path.with_suffix(".nc")

        canonical_path = output_dir / (out_path.stem + ".canonical.json")
        preview_path = output_dir / (out_path.stem + ".preview.png")

        self.write_gcode_from_strokes_mm(strokes_mm, out_path)
        self.save_canonical_inksight_json(strokes_mm, canonical_path)

        twin_suffix = ".gcode" if out_path.suffix.lower() == ".nc" else ".nc"
        twin_path = out_path.with_suffix(twin_suffix)
        shutil.copyfile(out_path, twin_path)

        canvas, map_pt, _, thickness, _ = self._fit_polys_to_canvas(strokes_mm)
        for path in strokes_mm:
            pts = np.array([map_pt(x, y) for x, y in path], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 0, 255), thickness, lineType=cv2.LINE_AA)
        cv2.imwrite(str(preview_path), canvas)

        self.log(f"✅ 输出笔画数：{len(strokes_mm)}")
        self.log(f"✅ 输出尺寸：{w_mm:.2f} mm × {h_mm:.2f} mm（已锁定到约 13×13 mm 外接框）")
        self.log(f"🎉 主输出文件：{out_path}")
        self.log(f"✅ 兼容副本：{twin_path}")
        self.log(f"✅ 规范化 JSON：{canonical_path}")
        self.log(f"✅ 预览图：{preview_path}")
        return str(out_path)

    def generate_gcode_from_inksight(self):
        if not self.inksight_strokes_pixels:
            messagebox.showwarning("提示", "请先导入 InkSight JSON")
            return
        try:
            base = Path(self.current_path).stem if self.current_path else "inksight_output"
            mode = self.inksight_last_mode or "inksight"
            self.generate_gcode_from_polys(self.inksight_strokes_pixels, f"{base}_{mode}.nc")
        except Exception as e:
            self.log(f"❌ InkSight G-code 生成失败: {e}")
            messagebox.showerror("错误", f"InkSight G-code 生成失败:\n{e}")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.png *.bmp *.jpeg")])
        if path:
            self.current_path = path
            self.img_label.config(text=f"已选择: {os.path.basename(path)}")
            self.log(f"加载成功: {path}")
            self.polys = []
            self.last_roi = None
            self.last_binary = None
            self.last_skeleton = None
            self.inksight_strokes_raw = []
            self.inksight_strokes_pixels = []
            self.inksight_json_var.set("")

    def process_current_image(self):
        if not self.current_path:
            messagebox.showwarning("提示", "请先选择图片！")
            return
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        self.generate_gcode(self.current_path)

    def start_camera(self):
        if self.preview_running:
            self.preview_running = False
            if self.cap:
                self.cap.release()
            self.log("摄像头关闭")
            return
        try:
            index = self.camera_index.get()
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                messagebox.showerror("错误", f"无法打开摄像头 {index}")
                return
            self.preview_running = True
            self.log(f"摄像头 {index} 打开成功")
            threading.Thread(target=self.camera_preview_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def camera_preview_loop(self):
        while self.preview_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_photo = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb).resize((640, 480))
                photo = ImageTk.PhotoImage(img_pil)
                self.canvas.create_image(0, 0, anchor="nw", image=photo)
                self.canvas.image = photo
            time.sleep(0.03)


if __name__ == "__main__":
    app = HandwritingGCodeApp()
