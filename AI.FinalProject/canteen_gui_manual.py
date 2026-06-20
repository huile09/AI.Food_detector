"""
Canteen Auto-Billing System
Hệ thống tự động tính tiền canteen
"""

import os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import threading
import time
from pathlib import Path
import sys

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Tọa độ các ô trên khay mẫu (reference size) - cập nhật mới
TRAY_REGIONS_REF = {
    "O1_tren_trai":  [375, 95, 770, 470],
    "O2_tren_phai":  [790, 100, 1170, 470],
    "O3_xa_phai":    [1195, 100, 1590, 460],
    "O4_duoi_trai":  [400, 520, 815, 1020],
    "O5_duoi_phai":  [985, 510, 1605, 1030],
}

REF_W, REF_H = 2048, 1152   # kích thước ảnh tham chiếu

# ─── CALIBRATION (chỉnh lệch tọa độ) ───────────────────────────────────────────
CALIB_PATH = os.path.join(os.path.dirname(__file__), "calib.json")
DEFAULT_CALIB = {"offset_x": 0, "offset_y": 0, "scale": 1.0}

def load_calib():
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
                return {**DEFAULT_CALIB, **d}
        except Exception:
            pass
    return dict(DEFAULT_CALIB)

def save_calib(calib):
    try:
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(calib, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Không lưu được calib: {e}")

CALIB = load_calib()

REGION_LABELS = {
    "O1_tren_trai":  "Ô 1 – Trên Trái",
    "O2_tren_phai":  "Ô 2 – Trên Giữa",
    "O3_xa_phai":    "Ô 3 – Trên Phải",
    "O4_duoi_trai":  "Ô 4 – Dưới Trái",
    "O5_duoi_phai":  "Ô 5 – Dưới Phải",
}

# Màu sắc ─ minimal dark theme
BG       = "#0f0f0f"
SURFACE  = "#1a1a1a"
BORDER   = "#2a2a2a"
ACCENT   = "#e8c547"       # vàng cơm / amber
TEXT     = "#f0f0f0"
SUBTEXT  = "#888888"
SUCCESS  = "#4caf7d"
ERROR    = "#e05c5c"
BOX_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]

FONT_MONO = ("Consolas", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_SM   = ("Segoe UI", 9)
FONT_LG   = ("Segoe UI", 14, "bold")
FONT_XL   = ("Segoe UI", 20, "bold")

MENU_PATH = os.path.join(os.path.dirname(__file__), "menu.json")
CROPS_DIR = os.path.join(os.path.dirname(__file__), "cropped_dishes")
os.makedirs(CROPS_DIR, exist_ok=True)

# Thư mục chứa file model (.h5 / .pt) – để cố định ở Downloads theo yêu cầu
MODELS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

# ─── LOAD MENU ─────────────────────────────────────────────────────────────────
def load_menu():
    try:
        with open(MENU_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["dishes"]
    except Exception:
        return {}

MENU = load_menu()

# Các món có thể chứa trứng -> đếm thủ công bằng bàn phím
EGG_DISH_KEYS   = {"Thịt kho trứng", "Trứng chiên", "Trứng chiên thịt"}
EGG_BASE_COUNT  = 1          # số trứng đã tính trong giá gốc
EGG_EXTRA_PRICE = 6000        # phụ thu mỗi trứng thêm
# Tất cả món đều tính tiền theo bảng giá
FREE_SIDE_KEYS = set()

# ─── MODEL LOADER ──────────────────────────────────────────────────────────────
CNN_MODEL  = None
# Thứ tự lớp PHẢI khớp chính xác với class_indices lúc train (KHÔNG sort lại):
# {'Canh chua có cá':0,'Canh chua không cá':1,'Canh rau':2,'Cá hú kho':3,'Cơm':4,
#  'Rau xào':5,'Sườn nướng':6,'Thịt kho':7,'Thịt kho trứng':8,'Trứng chiên':9,
#  'Trứng chiên thịt':10,'Đậu hũ sốt cà':11}
CLASS_NAMES = [
    "Canh chua có cá",
    "Canh chua không cá",
    "Canh rau",
    "Cá hú kho",
    "Cơm",
    "Rau xào",
    "Sườn nướng",
    "Thịt kho",
    "Thịt kho trứng",
    "Trứng chiên",
    "Trứng chiên thịt",
    "Đậu hũ sốt cà",
]
CNN_LOAD_ERROR  = None
def try_load_models():
    global CNN_MODEL, CNN_LOAD_ERROR
    # CNN — ưu tiên file .keras (đã convert), fallback .h5
    cnn_path_keras = os.path.join(MODELS_DIR, "best_food_model6.keras")
    cnn_path_h5    = os.path.join(MODELS_DIR, "best_food_model6.h5")
    cnn_path = cnn_path_keras if os.path.exists(cnn_path_keras) else cnn_path_h5
    if os.path.exists(cnn_path):
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import Dense

            # ── PATCH: Keras 3.x legacy .h5 loader đưa thêm 'quantization_config'
            # vào config của Dense, nhưng Dense.__init__ ở vài bản Keras 3 không
            # nhận tham số này -> bỏ nó đi trước khi khởi tạo layer.
            if not getattr(Dense, "_qc_patched", False):
                _orig_from_config = Dense.from_config.__func__

                @classmethod
                def _patched_from_config(cls, config):
                    config = dict(config)
                    config.pop("quantization_config", None)
                    return _orig_from_config(cls, config)

                Dense.from_config = _patched_from_config
                Dense._qc_patched = True

            CNN_MODEL = tf.keras.models.load_model(cnn_path)
            out_dim = CNN_MODEL.output_shape[-1]
            if out_dim != len(CLASS_NAMES):
                CNN_LOAD_ERROR = (f"Model có {out_dim} lớp nhưng menu.json có "
                                  f"{len(CLASS_NAMES)} món -> kết quả sẽ SAI. "
                                  f"Hãy sửa menu.json cho khớp số lớp đã train.")
                print(f"[WARN] {CNN_LOAD_ERROR}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            CNN_LOAD_ERROR = f"Không load được CNN model: {e}"
            print(f"[WARN] {CNN_LOAD_ERROR}")
    else:
        CNN_LOAD_ERROR = f"Không thấy {cnn_path_h5} -> đang chạy DEMO (random)."

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def scale_regions(img_w, img_h, calib=None):
    """Scale tọa độ tham chiếu sang kích thước ảnh thực, có áp dụng hiệu chỉnh
    lệch (offset_x, offset_y theo px tham chiếu) và scale (co/giãn quanh tâm)."""
    if calib is None:
        calib = CALIB
    sx = img_w / REF_W
    sy = img_h / REF_H
    cx, cy = REF_W / 2, REF_H / 2
    sc = calib.get("scale", 1.0)
    ox = calib.get("offset_x", 0)
    oy = calib.get("offset_y", 0)
    out = {}
    for key, (x1, y1, x2, y2) in TRAY_REGIONS_REF.items():
        nx1 = cx + (x1 - cx) * sc + ox
        nx2 = cx + (x2 - cx) * sc + ox
        ny1 = cy + (y1 - cy) * sc + oy
        ny2 = cy + (y2 - cy) * sc + oy
        out[key] = [int(nx1*sx), int(ny1*sy), int(nx2*sx), int(ny2*sy)]
    return out

def crop_regions(img_np):
    """Crop từng ô theo tọa độ đã scale."""
    h, w = img_np.shape[:2]
    regions = scale_regions(w, h)
    crops = {}
    for key, (x1, y1, x2, y2) in regions.items():
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img_np[y1:y2, x1:x2]
        crops[key] = crop
    return crops, regions



def predict_dish(crop_np):
    """Nhận dạng món ăn từ crop (crop_np là ảnh BGR từ OpenCV).
    Trả về (dish_key, confidence)."""
    if CNN_MODEL is not None:
        try:
            # OpenCV đọc ảnh ở dạng BGR, nhưng model Keras (ImageDataGenerator/PIL)
            # luôn train trên RGB -> PHẢI chuyển màu trước khi predict.
            rgb = cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (128, 128))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, 0)
            preds = CNN_MODEL.predict(img, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            if idx < len(CLASS_NAMES):
                return CLASS_NAMES[idx], conf
        except Exception as e:
            print(f"[CNN error] {e}")
    # Fallback: random demo khi không có model
    import random
    key = random.choice(CLASS_NAMES)
    return key, random.uniform(0.65, 0.95)

def draw_annotations(img_pil, regions, results):
    """Vẽ bounding box + nhãn lên ảnh."""
    draw = ImageDraw.Draw(img_pil, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", max(12, img_pil.width // 60))
        font_sm = ImageFont.truetype("arial.ttf", max(10, img_pil.width // 80))
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for i, (key, (x1, y1, x2, y2)) in enumerate(regions.items()):
        color_hex = BOX_COLORS[i % len(BOX_COLORS)]
        r, g, b = tuple(int(color_hex.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
        # box fill
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 220), width=3)
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 25))

        if key in results:
            dish_key, conf, egg_count, egg_extra, is_free_side = results[key]
            dish_info = MENU.get(dish_key, {})
            label = dish_info.get("name", dish_key)
            if is_free_side:
                price_txt = "kèm"
            else:
                price_txt = f"{(dish_info.get('price', 0) + egg_extra)//1000}k"
            txt = f"{label}\n{price_txt}  ({conf*100:.0f}%)"
            if egg_count is not None:
                txt += f"\n🥚x{egg_count}"
            # label bg
            box_h = 36 if egg_count is None else 52
            lx, ly = x1 + 6, y1 + 6
            draw.rectangle([lx-4, ly-4, lx+160, ly+box_h], fill=(0, 0, 0, 160))
            draw.text((lx, ly), txt, fill=(r, g, b, 255), font=font)

    return img_pil

# ─── MAIN APP ──────────────────────────────────────────────────────────────────
class CanteenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Canteen Auto-Billing  |  IMAGE RECOGNITING CHALLENGE 2025")
        self.configure(bg=BG)
        self.minsize(1320, 800)
        self.geometry("1500x900")
        self.resizable(True, True)

        self.current_img_np  = None   # numpy BGR
        self.results         = {}     # {region_key: (dish_key, conf)}
        self.cam_active      = False
        self.cam_thread      = None
        self.cap             = None
        self._photo_cache    = []
        self.cam_index       = 0      # camera index hiện tại

        self._build_ui()
        self._set_status("Sẵn sàng – chụp ảnh hoặc nhập ảnh từ máy tính", SUBTEXT)

        # Load models in background
        t = threading.Thread(target=self._bg_load_models, daemon=True)
        t.start()

    # ── UI BUILD ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(18, 0))
        tk.Label(hdr, text="🍱", bg=BG, font=("Segoe UI Emoji", 22)).pack(side="left")
        tk.Label(hdr, text=" CANTEEN CHECKOUT", bg=BG, fg=ACCENT,
                 font=FONT_XL).pack(side="left", padx=8)
        self.lbl_status = tk.Label(hdr, text="", bg=BG, fg=SUBTEXT, font=FONT_SM)
        self.lbl_status.pack(side="right")

        div = tk.Frame(self, bg=BORDER, height=1)
        div.pack(fill="x", padx=20, pady=10)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=0)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=320)
        body.rowconfigure(0, weight=1)

        # ── Left: image panel ──
        left = tk.Frame(body, bg=SURFACE, bd=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        toolbar = tk.Frame(left, bg=SURFACE)
        toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=10)

        self.btn_camera = self._btn(toolbar, "📷  Camera", self._toggle_camera)
        self.btn_camera.pack(side="left", padx=(0, 6))
        self.btn_connect = self._btn(toolbar, "🔌  Kết nối", self._show_connect_dialog)
        self.btn_connect.pack(side="left", padx=(0, 6))
        self.btn_connect.pack_forget()   # ẩn cho đến khi camera bật
        self._btn(toolbar, "🖼  Nhập ảnh", self._import_image).pack(side="left", padx=(0, 6))
        self._btn(toolbar, "🔍  Nhận diện", self._run_recognition, accent=True).pack(side="left", padx=(0, 6))
        self._btn(toolbar, "⚙  Hiệu chỉnh", self._toggle_calib).pack(side="left", padx=(0, 6))
        self._btn(toolbar, "↺  Xóa", self._clear).pack(side="right")

        # ── Calibration panel (hidden by default) ──
        self.calib_frame = tk.Frame(left, bg=SURFACE)
        self._build_calib_panel(self.calib_frame)
        # not gridded yet -> hidden

        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.lbl_hint = tk.Label(left, text="Chụp ảnh hoặc nhập ảnh khay cơm",
                                  bg=SURFACE, fg=SUBTEXT, font=FONT_SM)
        self.lbl_hint.grid(row=3, column=0, pady=(0, 6))

        # ── Right: bill panel ──
        right = tk.Frame(body, bg=SURFACE, width=320)
        right.grid(row=0, column=1, sticky="nsew", pady=0)
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        tk.Label(right, text="HÓA ĐƠN", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).grid(row=0, column=0,
                 sticky="w", padx=14, pady=(14, 4))

        bill_frame = tk.Frame(right, bg=SURFACE)
        bill_frame.grid(row=1, column=0, sticky="nsew", padx=14)

        self.bill_rows = []       # list of (label_frame) per dish
        self.egg_vars = {}
        self._build_bill_rows(bill_frame)

        # Divider
        tk.Frame(right, bg=BORDER, height=1).grid(row=2, column=0,
                  sticky="ew", padx=14, pady=8)

        # Total
        tot_f = tk.Frame(right, bg=SURFACE)
        tot_f.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        tk.Label(tot_f, text="TỔNG", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_total = tk.Label(tot_f, text="0 đ", bg=SURFACE, fg=ACCENT,
                                   font=("Segoe UI", 16, "bold"))
        self.lbl_total.pack(side="right")

        # Log
        tk.Frame(right, bg=BORDER, height=1).grid(row=4, column=0,
                  sticky="ew", padx=14, pady=(0, 6))
        tk.Label(right, text="LOG", bg=SURFACE, fg=SUBTEXT,
                 font=("Segoe UI", 8)).grid(row=5, column=0, sticky="w", padx=14)
        self.log_text = tk.Text(right, bg="#111111", fg=SUBTEXT, font=FONT_MONO,
                                 height=8, bd=0, relief="flat", wrap="word",
                                 insertbackground=SUBTEXT)
        self.log_text.grid(row=6, column=0, sticky="nsew", padx=14, pady=(2, 14))
        right.rowconfigure(6, weight=1)

    def _build_bill_rows(self, parent):
        for i, (key, region_label) in enumerate(REGION_LABELS.items()):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            row_f = tk.Frame(parent, bg=SURFACE)
            row_f.pack(fill="x", pady=3)

            dot = tk.Label(row_f, text="●", bg=SURFACE, fg=color, font=("Segoe UI", 8))
            dot.pack(side="left", padx=(0, 6))

            info_f = tk.Frame(row_f, bg=SURFACE)
            info_f.pack(side="left", fill="x", expand=True)

            lbl_slot = tk.Label(info_f, text=region_label, bg=SURFACE,
                                 fg=SUBTEXT, font=FONT_SM)
            lbl_slot.pack(anchor="w")

            lbl_dish = tk.Label(info_f, text="—", bg=SURFACE, fg=TEXT, font=FONT_BODY)
            lbl_dish.pack(anchor="w")

            lbl_egg = tk.Label(info_f, text="", bg=SURFACE, fg=SUBTEXT, font=("Segoe UI", 8))
            lbl_egg.pack(anchor="w")

            lbl_price = tk.Label(row_f, text="", bg=SURFACE, fg=ACCENT,
                                  font=("Segoe UI", 10, "bold"))
            lbl_price.pack(side="right")

            egg_var = tk.IntVar(value=EGG_BASE_COUNT)
            egg_spin = tk.Spinbox(row_f, from_=0, to=10, width=3,
                                   textvariable=egg_var, font=("Segoe UI", 9),
                                   bg=BORDER, fg=TEXT, buttonbackground=BORDER,
                                   relief="flat", justify="center",
                                   command=lambda k=key: self._on_egg_change(k))
            egg_var.trace_add("write", lambda *a, k=key: self._on_egg_change(k))

            self.bill_rows.append({
                "key": key, "lbl_dish": lbl_dish, "lbl_egg": lbl_egg,
                "lbl_price": lbl_price, "egg_var": egg_var, "egg_spin": egg_spin
            })
            self.egg_vars[key] = egg_var

    def _build_calib_panel(self, parent):
        parent.configure(bg=SURFACE)
        tk.Label(parent, text="HIỆU CHỈNH VỊ TRÍ Ô (lệch tọa độ)", bg=SURFACE,
                 fg=SUBTEXT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(2, 4))

        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=4)

        def make_slider(label, frm, to, key, resolution=1):
            f = tk.Frame(row, bg=SURFACE)
            f.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(f, text=label, bg=SURFACE, fg=SUBTEXT, font=FONT_SM).pack(anchor="w")
            var = tk.DoubleVar(value=CALIB.get(key, DEFAULT_CALIB[key]))
            s = tk.Scale(f, from_=frm, to=to, orient="horizontal", variable=var,
                          resolution=resolution, bg=SURFACE, fg=TEXT,
                          troughcolor=BORDER, highlightthickness=0,
                          activebackground=ACCENT, font=FONT_SM,
                          command=lambda v, k=key, vv=var: self._on_calib_change(k, vv))
            s.pack(fill="x")
            return var

        self.calib_vars = {}
        self.calib_vars["offset_x"] = make_slider("Lệch ngang (X)", -400, 400, "offset_x")
        self.calib_vars["offset_y"] = make_slider("Lệch dọc (Y)",   -400, 400, "offset_y")
        self.calib_vars["scale"]    = make_slider("Tỉ lệ (Scale)",  0.5, 1.5, "scale", resolution=0.01)

        btn_row = tk.Frame(parent, bg=SURFACE)
        btn_row.pack(fill="x", padx=4, pady=(2, 6))
        self._btn(btn_row, "💾 Lưu hiệu chỉnh", self._save_calib).pack(side="left", padx=(0, 6))
        self._btn(btn_row, "↺ Mặc định", self._reset_calib).pack(side="left")

    def _toggle_calib(self):
        if self.calib_frame.winfo_ismapped():
            self.calib_frame.grid_remove()
        else:
            self.calib_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))

    def _on_calib_change(self, key, var):
        try:
            val = var.get()
        except Exception:
            return
        CALIB[key] = val
        self._redraw_overlay()

    def _save_calib(self):
        save_calib(CALIB)
        self._set_status("Đã lưu hiệu chỉnh vào calib.json", SUCCESS)

    def _reset_calib(self):
        CALIB.update(DEFAULT_CALIB)
        for k, var in self.calib_vars.items():
            var.set(DEFAULT_CALIB[k])
        self._redraw_overlay()

    def _redraw_overlay(self):
        """Vẽ lại khung ô (và kết quả nếu có) trên ảnh hiện tại theo CALIB mới."""
        if self.current_img_np is None or self.cam_active:
            return
        h, w = self.current_img_np.shape[:2]
        regions = scale_regions(w, h)
        rgb = cv2.cvtColor(self.current_img_np, cv2.COLOR_BGR2RGB)
        self._show_image(rgb, regions, self.results if self.results else None)

    def _btn(self, parent, text, cmd, accent=False):
        fg   = BG if accent else TEXT
        bg   = ACCENT if accent else BORDER
        abg  = "#f0d050" if accent else "#3a3a3a"
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
                      font=FONT_SM, relief="flat", bd=0, padx=12, pady=6,
                      cursor="hand2")
        return b

    # ── STATUS / LOG ────────────────────────────────────────────────────────────
    def _set_status(self, msg, color=TEXT):
        self.lbl_status.config(text=msg, fg=color)

    def _log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    # ── MODEL LOAD ──────────────────────────────────────────────────────────────
    def _bg_load_models(self):
        self._set_status("Đang tải model…", SUBTEXT)
        try_load_models()
        status = []
        if CNN_MODEL:  status.append("CNN ✓")
        if not status: status.append("Demo mode (no model)")
        msg = "  ".join(status)
        color = SUCCESS if CNN_MODEL else SUBTEXT
        if CNN_LOAD_ERROR:
            msg = f"{msg}  ⚠ {CNN_LOAD_ERROR}"
            color = ERROR if CNN_MODEL is None else "#e0a05c"
            self._log(f"⚠ {CNN_LOAD_ERROR}")
            if CNN_MODEL is not None:
                self._log(f"   -> Số lớp model: {CNN_MODEL.output_shape[-1]}, "
                          f"số món menu.json: {len(CLASS_NAMES)}")
                self._log(f"   -> Thứ tự lớp (alphabet): {CLASS_NAMES}")
        self.after(0, lambda: self._set_status(msg, color))

    # ── IMAGE DISPLAY ───────────────────────────────────────────────────────────
    def _show_image(self, img_np_rgb, regions=None, results=None):
        """Hiển thị ảnh lên canvas, scale fit, vẽ overlay nếu có."""
        pil = Image.fromarray(img_np_rgb)
        if regions and results:
            pil = draw_annotations(pil, regions, results)

        cw = self.canvas.winfo_width()  or 700
        ch = self.canvas.winfo_height() or 500
        pil.thumbnail((cw, ch), Image.LANCZOS)
        ph = ImageTk.PhotoImage(pil)
        self._photo_cache = [ph]
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=ph, anchor="center")

    def _on_canvas_resize(self, event):
        if self.current_img_np is not None and not self.cam_active:
            rgb = cv2.cvtColor(self.current_img_np, cv2.COLOR_BGR2RGB)
            h, w = self.current_img_np.shape[:2]
            regions = scale_regions(w, h)
            self._show_image(rgb, regions if self.results else None,
                             self.results or None)

    # ── CAMERA ──────────────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if self.cam_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self, cam_index=None):
        if cam_index is not None:
            self.cam_index = cam_index
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", f"Không mở được camera (index {self.cam_index})!")
            return
        self.cam_active = True
        self.lbl_hint.config(text=f"Camera [{self.cam_index}] đang chạy – nhấn 'Nhận diện' để chụp")
        self._set_status(f"Camera [{self.cam_index}] ON", SUCCESS)
        self.btn_camera.config(text="⏹  Tắt Camera")
        self.btn_connect.pack(side="left", padx=(0, 6))   # hiện nút kết nối
        self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self.cam_thread.start()

    def _stop_camera(self):
        self.cam_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._set_status("Camera OFF", SUBTEXT)
        self.lbl_hint.config(text="Chụp ảnh hoặc nhập ảnh khay cơm")
        self.btn_camera.config(text="📷  Camera")
        self.btn_connect.pack_forget()   # ẩn nút kết nối

    def _show_connect_dialog(self):
        """Hiển thị dialog chọn camera index để kết nối lại."""
        dlg = tk.Toplevel(self)
        dlg.title("Kết nối Camera")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()

        # Center dialog
        dlg.geometry("300x180")
        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - 300) // 2
        y = self.winfo_y() + (self.winfo_height() - 180) // 2
        dlg.geometry(f"+{x}+{y}")

        tk.Label(dlg, text="Chọn camera để kết nối",
                 bg=BG, fg=TEXT, font=FONT_BODY).pack(pady=(18, 6))

        cam_var = tk.IntVar(value=self.cam_index)

        idx_frame = tk.Frame(dlg, bg=BG)
        idx_frame.pack(pady=4)
        tk.Label(idx_frame, text="Camera index:", bg=BG, fg=SUBTEXT,
                 font=FONT_SM).pack(side="left", padx=(0, 8))
        spin = tk.Spinbox(idx_frame, from_=0, to=9, width=4,
                          textvariable=cam_var, font=FONT_BODY,
                          bg=BORDER, fg=TEXT, buttonbackground=BORDER,
                          relief="flat", justify="center")
        spin.pack(side="left")

        status_lbl = tk.Label(dlg, text="", bg=BG, fg=SUBTEXT, font=FONT_SM)
        status_lbl.pack(pady=(2, 0))

        def do_connect():
            idx = cam_var.get()
            status_lbl.config(text=f"Đang kết nối camera {idx}…", fg=ACCENT)
            dlg.update()
            # Dừng camera cũ nếu đang chạy
            self.cam_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            time.sleep(0.1)
            self._start_camera(cam_index=idx)
            if self.cam_active:
                dlg.destroy()
            else:
                status_lbl.config(text=f"Không tìm thấy camera {idx}", fg=ERROR)

        btn_f = tk.Frame(dlg, bg=BG)
        btn_f.pack(pady=12)
        self._btn(btn_f, "✔  Kết nối", do_connect, accent=True).pack(side="left", padx=6)
        self._btn(btn_f, "Hủy", dlg.destroy).pack(side="left", padx=6)

    def _cam_loop(self):
        while self.cam_active:
            ret, frame = self.cap.read()
            if ret:
                self.current_img_np = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.after(0, lambda r=rgb: self._show_image(r))
            time.sleep(0.03)

    # ── IMPORT IMAGE ─────────────────────────────────────────────────────────────
    def _import_image(self):
        if self.cam_active:
            self._stop_camera()
        path = filedialog.askopenfilename(
            title="Chọn ảnh khay cơm",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi", f"Không đọc được ảnh:\n{path}")
            return
        self.current_img_np = img
        self.results = {}
        h, w = img.shape[:2]
        regions = scale_regions(w, h)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._show_image(rgb, regions)
        self._reset_bill()
        self._log(f"Đã nhập: {os.path.basename(path)}  ({w}×{h})")
        self._set_status(f"Ảnh sẵn sàng – nhấn Nhận diện", TEXT)
        self.lbl_hint.config(text=f"{os.path.basename(path)}  ({w}×{h}px)")

    # ── RECOGNITION ──────────────────────────────────────────────────────────────
    def _run_recognition(self):
        if self.current_img_np is None:
            if self.cam_active:
                # capture current frame
                pass
            else:
                messagebox.showinfo("Thông báo", "Chưa có ảnh. Hãy chụp hoặc nhập ảnh trước.")
                return

        img = self.current_img_np.copy()
        self._stop_camera()

        self._set_status("Đang nhận diện…", ACCENT)
        self._log("─" * 40)
        self._log("▶ Bắt đầu nhận diện…")

        def worker():
            h, w = img.shape[:2]
            regions = scale_regions(w, h)
            crops, _ = crop_regions(img)
            results = {}
            lines = []
            charge_total = 0   # tổng tiền theo combo (không cộng món "đi kèm")

            for key, crop in crops.items():
                if crop.size == 0:
                    continue
                # Save crop
                crop_path = os.path.join(CROPS_DIR, f"{key}.jpg")
                cv2.imwrite(crop_path, crop)

                dish_key, conf = predict_dish(crop)
                info = MENU.get(dish_key, {})
                name = info.get("name", dish_key)
                price = info.get("price", 0)
                is_free_side = dish_key in FREE_SIDE_KEYS

                egg_count = None
                egg_extra = 0
                if dish_key in EGG_DISH_KEYS:
                    if dish_key == "Thịt kho trứng":
                       egg_count = self.egg_vars.get(key, tk.IntVar(value=EGG_BASE_COUNT)).get()
                    else:
                            egg_count = self.egg_vars.get(key, tk.IntVar(value=EGG_BASE_COUNT)).get()
                    extra_eggs = max(0, egg_count - EGG_BASE_COUNT)
                    egg_extra = extra_eggs * EGG_EXTRA_PRICE

                results[key] = (dish_key, conf, egg_count, egg_extra, is_free_side)

                if not is_free_side:
                    charge_total += price + egg_extra

                line = f"  {REGION_LABELS.get(key, key):18s}  {name:20s}"
                if is_free_side:
                    line += f"  (kèm)      ({conf*100:.0f}%)"
                else:
                    line += f"  {price:>7,}đ  ({conf*100:.0f}%)"
                if egg_count is not None:
                    line += f"   +🥚x{egg_count}"
                    if egg_extra > 0:
                        line += f" (+{egg_extra:,}đ)"
                lines.append(line)

            total = charge_total
            self.results = results
            self.after(0, lambda: self._update_results(img, regions, results, lines, total))

        threading.Thread(target=worker, daemon=True).start()

    def _update_results(self, img, regions, results, lines, total):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._show_image(rgb, regions, results)

        for row in self.bill_rows:
            key = row["key"]
            if key in results:
                dish_key, conf, egg_count, egg_extra, is_free_side = results[key]
                info = MENU.get(dish_key, {})
                name = info.get("name", "?")
                price = info.get("price", 0)
                row["lbl_dish"].config(text=f"{name}  ({conf*100:.0f}%)", fg=TEXT)
                if dish_key in EGG_DISH_KEYS:
                    row["egg_var"].set(egg_count if egg_count is not None else EGG_BASE_COUNT)
                    row["egg_spin"].pack(side="right", padx=(4, 6))
                    egg_txt = f"🥚 x{row['egg_var'].get()}"
                    if egg_extra > 0:
                        egg_txt += f"  (+{egg_extra:,}đ)"
                    row["lbl_egg"].config(text=egg_txt)
                else:
                    row["egg_spin"].pack_forget()
                    row["lbl_egg"].config(text="")
                if is_free_side:
                    row["lbl_price"].config(text="kèm", fg=SUBTEXT)
                else:
                    row["lbl_price"].config(text=f"{price+egg_extra:,}đ", fg=ACCENT)
            else:
                row["lbl_dish"].config(text="—", fg=SUBTEXT)
                row["egg_spin"].pack_forget()
                row["lbl_egg"].config(text="")
                row["lbl_price"].config(text="", fg=ACCENT)

        self.lbl_total.config(text=f"{total:,} đ")

        for l in lines:
            self._log(l)
        self._log(f"{'TỔNG HOÁ ĐƠN':>50s}  {total:>7,}đ")
        self._log(f"Ảnh crop lưu tại: {CROPS_DIR}")
        self._set_status(f"Nhận diện xong · Tổng: {total:,}đ", SUCCESS)
        self.lbl_hint.config(text=f"✓ Nhận diện xong – {len(results)} ô")

        # Console output
        print("\n" + "="*60)
        print("  HOÁ ĐƠN CANTEEN")
        print("="*60)
        for l in lines:
            print(l)
        print("-"*60)
        print(f"  TỔNG CỘNG: {total:,} đồng")
        print("="*60)

    def _on_egg_change(self, key):
        if key not in self.results:
            return
        try:
            egg_count = self.egg_vars[key].get()
        except Exception:
            return
        dish_key, conf, _, _, is_free_side = self.results[key]
        if dish_key not in EGG_DISH_KEYS:
            return
        extra_eggs = max(0, egg_count - EGG_BASE_COUNT)
        egg_extra = extra_eggs * EGG_EXTRA_PRICE
        self.results[key] = (dish_key, conf, egg_count, egg_extra, is_free_side)
        info = MENU.get(dish_key, {})
        price = info.get("price", 0)
        for row in self.bill_rows:
            if row["key"] == key:
                egg_txt = f"🥚 x{egg_count}"
                if egg_extra > 0:
                    egg_txt += f"  (+{egg_extra:,}đ)"
                row["lbl_egg"].config(text=egg_txt)
                if not is_free_side:
                    row["lbl_price"].config(text=f"{price+egg_extra:,}đ")
        self._recalc_total()
        self._redraw_overlay()
        self._log(f"  ✎ Cập nhật số trứng [{REGION_LABELS.get(key, key)}]: {egg_count}")

    def _recalc_total(self):
        total = 0
        for dish_key, conf, egg_count, egg_extra, is_free_side in self.results.values():
            if not is_free_side:
                info = MENU.get(dish_key, {})
                total += info.get("price", 0) + egg_extra
        self.lbl_total.config(text=f"{total:,} đ")
        return total

    def _reset_bill(self):
        for row in self.bill_rows:
            row["lbl_dish"].config(text="—", fg=SUBTEXT)
            row["lbl_egg"].config(text="")
            row["lbl_price"].config(text="")
            row["egg_spin"].pack_forget()
            row["egg_var"].set(EGG_BASE_COUNT)
        self.lbl_total.config(text="0 đ")

    def _clear(self):
        if self.cam_active:
            self._stop_camera()
        self.current_img_np = None
        self.results = {}
        self.canvas.delete("all")
        self._reset_bill()
        self.log_text.delete("1.0", "end")
        self.lbl_hint.config(text="Chụp ảnh hoặc nhập ảnh khay cơm")
        self._set_status("Đã xóa", SUBTEXT)

    def on_close(self):
        self._stop_camera()
        self.destroy()


# ─── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = CanteenApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()