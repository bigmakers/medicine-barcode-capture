#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医薬品バーコード検知・自動撮影システム
CZUR Shine 書画カメラ対応
"""

import os
import sys
import queue
import time
import threading
import sqlite3
import subprocess
from datetime import datetime, date
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3

# --- バーコードデコーダの検出 (zxing-cpp を優先: GS1 DataBar 対応) ---
try:
    import zxingcpp  # type: ignore

    HAS_ZXINGCPP = True
except ImportError:
    HAS_ZXINGCPP = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode

    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

# GS1 DataBar 系フォーマット (zxing-cpp の BarcodeFormat 名)
GS1_DATABAR_FORMATS = {
    "DataBar",
    "DataBarExpanded",
    "DataBarLimited",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  定数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RING_BUFFER_SEC = 2
CAPTURE_INTERVAL_SEC = 2
CAPTURE_COUNT = 10
PREVIEW_MAX_W = 960
PREVIEW_MAX_H = 540
DEFAULT_BASE_DIR = str(Path.home() / "MedicineCaptures")
DB_NAME = "captures.db"

# ── 動体検知パラメータ ──
MOTION_AREA_RATIO = 0.005       # 画面面積のこの割合以上が変化 → 動体あり
FRAME_DIFF_THRESHOLD = 3.0      # フレーム間差分の平均がこれ以下 → 静止
STABLE_REQUIRED_FRAMES = 20     # 連続これだけ静止フレームが続けば「安定」
MOTION_COOLDOWN_SEC = 3.0       # 撮影完了後、再検知を抑制する秒数

# ── 動体検知ステートマシン ──
STATE_IDLE = "IDLE"             # 待機中（背景のみ、変化なし）
STATE_MOTION = "MOTION"         # 動体検知中（何かが動いている）
STATE_STABILIZING = "STABILIZING"  # 動体が止まりつつある
STATE_READY = "READY"           # 安定 → バーコードスキャン実行


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ユーティリティ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def decode_barcodes(frame):
    """利用可能なデコーダでバーコードを検出する。zxing-cpp 優先（GS1 DataBar 対応）。"""
    results = []

    if HAS_ZXINGCPP:
        for d in zxingcpp.read_barcodes(frame):
            pos = d.position
            fmt = str(d.format)
            results.append(
                {
                    "data": d.text,
                    "type": fmt,
                    "is_gs1_databar": any(f in fmt for f in GS1_DATABAR_FORMATS),
                    "polygon": [
                        (pos.top_left.x, pos.top_left.y),
                        (pos.top_right.x, pos.top_right.y),
                        (pos.bottom_right.x, pos.bottom_right.y),
                        (pos.bottom_left.x, pos.bottom_left.y),
                    ],
                    "rect": None,
                }
            )
    elif HAS_PYZBAR:
        for d in pyzbar_decode(frame):
            results.append(
                {
                    "data": d.data.decode("utf-8", errors="replace"),
                    "type": d.type,
                    "is_gs1_databar": False,
                    "polygon": [(p.x, p.y) for p in d.polygon] if d.polygon else None,
                    "rect": (d.rect.left, d.rect.top, d.rect.width, d.rect.height),
                }
            )

    return results


def save_frame_as_jpg(filepath: str, frame, quality: int = 95) -> bool:
    """フレームをJPEGで保存する。cv2.imwrite は非ASCII パスで失敗するため imencode を使用。"""
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            print(f"エラー: JPEG エンコード失敗 — {filepath}")
            return False
        Path(filepath).write_bytes(buf.tobytes())
        print(f"保存: {filepath}")
        return True
    except Exception as e:
        print(f"エラー: 保存失敗 — {filepath}: {e}")
        return False


def open_file(filepath: str):
    """OS既定のアプリケーションでファイルを開く。"""
    if not os.path.exists(filepath):
        return
    if sys.platform == "win32":
        os.startfile(filepath)
    elif sys.platform == "darwin":
        subprocess.run(["open", filepath], check=False)
    else:
        subprocess.run(["xdg-open", filepath], check=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  データベース
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Database:
    """SQLite による撮影インデックス管理。"""

    def __init__(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        self.db_path = os.path.join(base_dir, DB_NAME)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS captures (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    barcode     TEXT    NOT NULL,
                    barcode_type TEXT,
                    filename    TEXT    NOT NULL,
                    filepath    TEXT    NOT NULL,
                    captured_at TEXT    NOT NULL,
                    session_id  TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_captured_at ON captures(captured_at)"
            )

    def insert(self, *, barcode, barcode_type, filename, filepath, captured_at, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO captures "
                "(barcode, barcode_type, filename, filepath, captured_at, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (barcode, barcode_type, filename, filepath, captured_at, session_id),
            )

    def search_by_date(self, target_date: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM captures WHERE captured_at LIKE ? ORDER BY captured_at DESC",
                (f"{target_date}%",),
            ).fetchall()

    def get_all_dates(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT substr(captured_at, 1, 10) AS d "
                "FROM captures ORDER BY d DESC"
            ).fetchall()
            return [r[0] for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  音声合成ワーカー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TTSWorker(threading.Thread):
    """バックグラウンドで音声合成を行うスレッド。"""

    def __init__(self):
        super().__init__(daemon=True)
        self._queue: queue.Queue = queue.Queue()

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
        except Exception:
            return
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def speak(self, text: str):
        self._queue.put(text)

    def shutdown(self):
        self._queue.put(None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  カメラスレッド
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CameraThread(threading.Thread):
    """カメラ映像取得・動体検知・バーコード検出を行うスレッド。

    ステートマシン:
        IDLE  ──(画面に変化)──▶  MOTION
        MOTION ──(動きが止まる)──▶  STABILIZING
        STABILIZING ──(一定フレーム静止継続)──▶  READY (バーコードスキャン)
        STABILIZING ──(再び動く)──▶  MOTION
        READY ──(スキャン完了)──▶  IDLE
    """

    def __init__(self, app: "App", camera_index: int = 0):
        super().__init__(daemon=True)
        self.app = app
        self.camera_index = camera_index
        self.running = False
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30
        self.ring_buffer: deque = deque(maxlen=RING_BUFFER_SEC * self.fps)

        # ── 動体検知 ──
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False,
        )
        self.prev_gray = None
        self.state = STATE_IDLE
        self.stable_count = 0
        self.motion_area_ratio = 0.0   # デバッグ表示用
        self.frame_diff = 0.0          # デバッグ表示用

        # ── 撮影制御 ──
        self.is_capturing = False
        self.capture_progress = 0
        self.dialog_shown = False
        self.cooldown_until = 0.0      # 撮影後の再検知抑制

    # ──────────────────────────────────
    #  メインループ
    # ──────────────────────────────────
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.app.after(0, lambda: self.app.set_status("エラー: カメラを開けませんでした"))
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = max(int(actual_fps), 1)
        self.ring_buffer = deque(maxlen=max(RING_BUFFER_SEC * self.fps, 10))

        self.running = True
        self.app.after(0, lambda: self.app.set_status("カメラ接続済み — 監視中"))

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            timestamp = datetime.now()
            self.ring_buffer.append((frame.copy(), timestamp))

            display = frame.copy()

            if not self.is_capturing and not self.dialog_shown:
                self._step_state_machine(frame, display)

            self._draw_overlay(display)
            self.app.after(0, lambda f=display: self.app.update_preview(f))

            time.sleep(1.0 / self.fps)

        if self.cap:
            self.cap.release()

    # ──────────────────────────────────
    #  ステートマシン
    # ──────────────────────────────────
    def _step_state_machine(self, frame, display):
        has_motion, is_still = self._analyze_motion(frame)

        # ── 撮影後クールダウン中 ──
        if time.time() < self.cooldown_until:
            self.state = STATE_IDLE
            return

        # ── IDLE: 変化待ち ──
        if self.state == STATE_IDLE:
            if has_motion:
                self.state = STATE_MOTION
                self.stable_count = 0

        # ── MOTION: 動いている ──
        elif self.state == STATE_MOTION:
            if is_still:
                self.stable_count += 1
                if self.stable_count >= STABLE_REQUIRED_FRAMES:
                    self.state = STATE_STABILIZING
            else:
                self.stable_count = 0

        # ── STABILIZING → READY: バーコードスキャン ──
        elif self.state == STATE_STABILIZING:
            if has_motion:
                # また動き出した → MOTION に戻る
                self.state = STATE_MOTION
                self.stable_count = 0
                return

            self.state = STATE_READY
            barcodes = decode_barcodes(frame)

            # バーコード枠を描画
            for bc in barcodes:
                if bc["polygon"]:
                    pts = np.array(bc["polygon"], dtype=np.int32)
                    cv2.polylines(display, [pts], True, (0, 255, 0), 3)
                elif bc["rect"]:
                    x, y, w, h = bc["rect"]
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
                label = f'{bc["type"]}: {bc["data"][:40]}'
                cv2.putText(
                    display, label, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            if barcodes:
                bc = barcodes[0]
                self.dialog_shown = True
                self.app.after(
                    0,
                    lambda d=bc["data"], t=bc["type"]: self.app.show_confirm_dialog(d, t),
                )
            else:
                # バーコードなし → IDLE に戻り次の動体を待つ
                self.state = STATE_IDLE

        # ── READY: ダイアログ表示中 or 処理待ち ──
        elif self.state == STATE_READY:
            pass  # dialog_shown フラグで制御される

    def _analyze_motion(self, frame):
        """背景差分 + フレーム間差分で動体を判定。

        Returns:
            has_motion: 背景と比較して有意な変化領域がある
            is_still:   直前フレームとの差が小さい（物体が静止）
        """
        # 背景差分 (MOG2) — 「何か新しいものがあるか」
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        fg_pixels = np.count_nonzero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        self.motion_area_ratio = fg_pixels / total_pixels
        has_motion = self.motion_area_ratio > MOTION_AREA_RATIO

        # フレーム間差分 — 「今この瞬間動いているか」
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return has_motion, False

        diff = cv2.absdiff(self.prev_gray, gray)
        self.frame_diff = float(np.mean(diff))
        self.prev_gray = gray

        is_still = self.frame_diff < FRAME_DIFF_THRESHOLD
        return has_motion, is_still

    # ──────────────────────────────────
    #  オーバーレイ描画
    # ──────────────────────────────────
    def _draw_overlay(self, display):
        h, w = display.shape[:2]

        if self.is_capturing:
            state_text = f"CAPTURING {self.capture_progress}/{CAPTURE_COUNT}"
            color = (0, 0, 255)
            if self.capture_progress > 0:
                bar_w = int(w * 0.5)
                bar_x = (w - bar_w) // 2
                bar_y = h - 40
                progress = self.capture_progress / CAPTURE_COUNT
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(
                    display, (bar_x, bar_y),
                    (bar_x + int(bar_w * progress), bar_y + 20),
                    (0, 200, 0), -1,
                )
                cv2.putText(
                    display, f"{self.capture_progress}/{CAPTURE_COUNT}",
                    (bar_x + bar_w + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
        else:
            # ステート表示
            state_colors = {
                STATE_IDLE: ((200, 200, 200), "IDLE"),
                STATE_MOTION: ((0, 165, 255), "MOTION DETECTED"),
                STATE_STABILIZING: ((0, 255, 255), "STABILIZING..."),
                STATE_READY: ((0, 255, 0), "READY - SCANNING"),
            }
            color, state_text = state_colors.get(self.state, ((255, 255, 255), self.state))

        cv2.putText(display, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # デバッグ情報
        if not self.is_capturing:
            info = f"BG:{self.motion_area_ratio:.4f}  DIFF:{self.frame_diff:.1f}  STABLE:{self.stable_count}"
            cv2.putText(display, info, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ──────────────────────────────────
    #  バッファアクセス
    # ──────────────────────────────────
    def get_buffer_frame(self):
        """リングバッファ最古フレーム（≒2秒前）を返す。"""
        if self.ring_buffer:
            return self.ring_buffer[0]
        return None, None

    def get_current_frame(self):
        """最新フレームを返す。"""
        if self.ring_buffer:
            return self.ring_buffer[-1]
        return None, None

    def stop(self):
        self.running = False

    def reset_for_next(self):
        """撮影完了後のリセット。クールダウン付きで IDLE に戻す。"""
        self.dialog_shown = False
        self.cooldown_until = time.time() + MOTION_COOLDOWN_SEC
        self.state = STATE_IDLE
        self.stable_count = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  メインアプリケーション
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("医薬品バーコード検知・自動撮影システム")
        self.geometry("1100x750")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.base_dir = ctk.StringVar(value=DEFAULT_BASE_DIR)
        self.camera_index = ctk.IntVar(value=0)

        self.camera_thread: CameraThread | None = None
        self.tts = TTSWorker()
        self.tts.start()
        self.db: Database | None = None
        self._photo_ref = None  # PhotoImage の参照保持（GC防止）

        self._build_ui()
        self._init_db()

    def _init_db(self):
        os.makedirs(self.base_dir.get(), exist_ok=True)
        self.db = Database(self.base_dir.get())

    # ────────────────────────────────────
    #  UI 構築
    # ────────────────────────────────────
    def _build_ui(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        self._build_camera_tab()
        self._build_search_tab()

        self.status_var = ctk.StringVar(value="待機中")
        ctk.CTkLabel(self, textvariable=self.status_var, anchor="w").pack(
            fill="x", padx=15, pady=(0, 8)
        )

    def _build_camera_tab(self):
        tab = self.tabview.add("カメラ")

        ctrl = ctk.CTkFrame(tab)
        ctrl.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(ctrl, text="カメラ番号:").pack(side="left", padx=(10, 2))
        ctk.CTkEntry(ctrl, textvariable=self.camera_index, width=50).pack(side="left", padx=2)

        self.btn_start = ctk.CTkButton(ctrl, text="▶ 開始", width=80, command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ctk.CTkButton(
            ctrl, text="■ 停止", width=80, command=self.stop_camera, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=5)

        ctk.CTkLabel(ctrl, text="保存先:").pack(side="left", padx=(20, 2))
        ctk.CTkEntry(ctrl, textvariable=self.base_dir, width=300).pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="参照", width=50, command=self._browse_dir).pack(
            side="left", padx=5
        )

        # プレビュー（tk.Label で高速な映像更新）
        self.preview = tk.Label(tab, bg="#1a1a1a", text="カメラ未接続", fg="gray")
        self.preview.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_search_tab(self):
        tab = self.tabview.add("検索")

        top = ctk.CTkFrame(tab)
        top.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(top, text="日付 (yyyy-mm-dd):").pack(side="left", padx=5)
        self.search_entry = ctk.CTkEntry(top, width=140)
        self.search_entry.insert(0, date.today().isoformat())
        self.search_entry.pack(side="left", padx=5)

        ctk.CTkButton(top, text="検索", width=80, command=self._do_search).pack(
            side="left", padx=5
        )
        ctk.CTkButton(top, text="日付一覧", width=80, command=self._show_dates).pack(
            side="left", padx=5
        )

        self.results_frame = ctk.CTkScrollableFrame(tab)
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)

    # ────────────────────────────────────
    #  カメラ制御
    # ────────────────────────────────────
    def start_camera(self):
        self._init_db()
        self.camera_thread = CameraThread(self, self.camera_index.get())
        self.camera_thread.start()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.set_status("カメラ起動中...")

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.preview.configure(image="", text="カメラ未接続")
        self._photo_ref = None
        self.set_status("停止")

    def set_status(self, text: str):
        self.status_var.set(text)

    def update_preview(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=photo, text="")
            self._photo_ref = photo
        except Exception:
            pass

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.base_dir.get())
        if d:
            self.base_dir.set(d)
            self._init_db()

    # ────────────────────────────────────
    #  確認ダイアログ & 撮影シーケンス
    # ────────────────────────────────────
    def show_confirm_dialog(self, barcode_data: str, barcode_type: str):
        dialog = ctk.CTkToplevel(self)
        dialog.title("バーコード検知")
        dialog.geometry("450x230")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 450) // 2
        y = self.winfo_y() + (self.winfo_height() - 230) // 2
        dialog.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dialog, text="バーコードを検知しました",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(20, 10))
        ctk.CTkLabel(dialog, text=f"種類: {barcode_type}").pack()
        display_data = barcode_data if len(barcode_data) <= 50 else barcode_data[:50] + "..."
        ctk.CTkLabel(dialog, text=f"データ: {display_data}").pack(pady=5)
        ctk.CTkLabel(dialog, text="登録しますか？", font=ctk.CTkFont(size=14)).pack(pady=5)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)

        def on_register():
            dialog.destroy()
            self._start_capture(barcode_data, barcode_type)

        def on_cancel():
            dialog.destroy()
            if self.camera_thread:
                self.camera_thread.reset_for_next()

        ctk.CTkButton(btn_frame, text="登録する", width=120, command=on_register).pack(
            side="left", padx=10
        )
        ctk.CTkButton(
            btn_frame, text="キャンセル", width=120, fg_color="gray", command=on_cancel
        ).pack(side="left", padx=10)

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

    def _start_capture(self, barcode_data: str, barcode_type: str):
        if self.camera_thread:
            self.camera_thread.is_capturing = True
            self.camera_thread.capture_progress = 0
        threading.Thread(
            target=self._capture_sequence,
            args=(barcode_data, barcode_type),
            daemon=True,
        ).start()

    def _capture_sequence(self, barcode_data: str, barcode_type: str):
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        date_dir = now.strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_dir.get(), date_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.after(0, lambda: self.set_status(f"撮影中 0/{CAPTURE_COUNT} — {barcode_data}"))
        saved_count = 0

        for i in range(CAPTURE_COUNT):
            if i == 0:
                # リングバッファの最古フレーム（≒2秒前）
                frame, ts = (
                    self.camera_thread.get_buffer_frame()
                    if self.camera_thread
                    else (None, None)
                )
                if frame is None and self.camera_thread:
                    frame, ts = self.camera_thread.get_current_frame()
            else:
                time.sleep(CAPTURE_INTERVAL_SEC)
                frame, ts = (
                    self.camera_thread.get_current_frame()
                    if self.camera_thread
                    else (None, None)
                )

            if frame is None:
                print(f"警告: フレーム {i + 1}/{CAPTURE_COUNT} が None — スキップ")
                continue

            ts = ts or datetime.now()
            filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{i + 1:02d}.jpg"
            filepath = os.path.join(save_dir, filename)

            if save_frame_as_jpg(filepath, frame):
                saved_count += 1
                self.db.insert(
                    barcode=barcode_data,
                    barcode_type=barcode_type,
                    filename=filename,
                    filepath=filepath,
                    captured_at=ts.strftime("%Y-%m-%d %H:%M:%S"),
                    session_id=session_id,
                )

            count = i + 1
            if self.camera_thread:
                self.camera_thread.capture_progress = count
            self.after(
                0,
                lambda c=count: self.set_status(f"撮影中 {c}/{CAPTURE_COUNT} — {barcode_data}"),
            )

        # 撮影完了 → クールダウン付きリセット
        if self.camera_thread:
            self.camera_thread.is_capturing = False
            self.camera_thread.capture_progress = 0
            self.camera_thread.reset_for_next()

        msg = f"撮影完了: {saved_count}/{CAPTURE_COUNT}枚保存 ({barcode_data})"
        self.after(0, lambda: self.set_status(msg))
        print(msg)
        print(f"保存先: {save_dir}")
        self.tts.speak(f"登録完了。{saved_count}枚保存しました。バーコード {barcode_data}")

    # ────────────────────────────────────
    #  検索
    # ────────────────────────────────────
    def _do_search(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

        target = self.search_entry.get().strip()
        if not target:
            return

        rows = self.db.search_by_date(target)

        if not rows:
            ctk.CTkLabel(
                self.results_frame, text="該当データがありません", text_color="gray"
            ).pack(pady=20)
            return

        ctk.CTkLabel(
            self.results_frame,
            text=f"{len(rows)} 件",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=5)

        for row in rows:
            rf = ctk.CTkFrame(self.results_frame)
            rf.pack(fill="x", padx=5, pady=2)

            info = (
                f"{row['captured_at']}  |  "
                f"{row['barcode_type']}: {row['barcode']}  |  "
                f"{row['filename']}"
            )
            ctk.CTkLabel(rf, text=info, anchor="w").pack(
                side="left", padx=10, pady=5, fill="x", expand=True
            )

            fp = row["filepath"]
            ctk.CTkButton(
                rf,
                text="開く",
                width=50,
                command=lambda p=fp: open_file(p),
            ).pack(side="right", padx=5, pady=5)

    def _show_dates(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

        dates = self.db.get_all_dates()
        if not dates:
            ctk.CTkLabel(
                self.results_frame, text="データがありません", text_color="gray"
            ).pack(pady=20)
            return

        ctk.CTkLabel(
            self.results_frame,
            text="撮影日一覧",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=5)

        for d in dates:
            ctk.CTkButton(
                self.results_frame,
                text=d,
                anchor="w",
                command=lambda dt=d: self._search_date(dt),
            ).pack(fill="x", padx=10, pady=2)

    def _search_date(self, dt: str):
        self.search_entry.delete(0, "end")
        self.search_entry.insert(0, dt)
        self._do_search()

    # ────────────────────────────────────
    #  終了処理
    # ────────────────────────────────────
    def on_closing(self):
        self.stop_camera()
        self.tts.shutdown()
        self.destroy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  エントリポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    if HAS_ZXINGCPP:
        print("バーコードデコーダ: zxing-cpp (GS1 DataBar 対応)")
    elif HAS_PYZBAR:
        print(
            "バーコードデコーダ: pyzbar (GS1 DataBar 非対応 — "
            "pip install zxing-cpp を推奨)"
        )
    else:
        print(
            "警告: バーコードデコーダ (zxing-cpp または pyzbar) が"
            "インストールされていません。バーコード検知は無効です。"
        )

    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
