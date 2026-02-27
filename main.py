#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医薬品バーコード検知・自動撮影システム
CZUR Shine 書画カメラ対応
"""

import os
import sys
import json
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
CAPTURE_INTERVAL_SEC = 1
CAPTURE_COUNT = 5
PREVIEW_MAX_W = 960
PREVIEW_MAX_H = 540
DEFAULT_BASE_DIR = str(Path.home() / "MedicineCaptures")
DB_NAME = "captures.db"
CONFIG_PATH = str(Path.home() / ".medicine_capture_config.json")

# ── 動体検知パラメータ ──
MOTION_AREA_RATIO = 0.005       # 画面面積のこの割合以上が変化 → 動体あり
MOTION_COOLDOWN_SEC = 3.0       # 撮影完了後、再検知を抑制する秒数

# ── 計数モードパラメータ ──
PILL_THRESH_DEFAULT = 60        # 二値化しきい値デフォルト
PILL_MIN_AREA = 300             # 輪郭面積の最小（ノイズ除去）
PILL_MAX_AREA = 50000           # 輪郭面積の最大（手等を除外）
PILL_BLUR_KSIZE = 7             # ガウシアンぼかしカーネル
PILL_CIRCULARITY_ROUND = 0.80   # 円形度がこれ以上 → 丸
PILL_ASPECT_CAPSULE = 2.0       # アスペクト比がこれ以上 → カプセル

# ── 動体検知ステート ──
STATE_IDLE = "IDLE"             # 待機中（背景のみ、変化なし）
STATE_TRIGGERED = "TRIGGERED"   # 動体検知 → ダイアログ表示中


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS barcode_registry (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    barcode    TEXT    NOT NULL UNIQUE,
                    name       TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                )
                """
            )

    def insert(self, *, barcode, barcode_type, filename, filepath, captured_at, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO captures "
                "(barcode, barcode_type, filename, filepath, captured_at, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (barcode, barcode_type, filename, filepath, captured_at, session_id),
            )

    # ── バーコード登録 ──
    def register_barcode(self, barcode: str, name: str):
        """バーコードと薬名を登録（既存なら更新）。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO barcode_registry (barcode, name, created_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(barcode) DO UPDATE SET name=excluded.name, created_at=excluded.created_at",
                (barcode, name, now),
            )

    def lookup_barcode(self, barcode: str):
        """バーコードから薬名を検索。未登録なら None を返す。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT name FROM barcode_registry WHERE barcode = ?", (barcode,)
            ).fetchone()
            return row[0] if row else None

    def get_all_registered(self):
        """登録済みバーコード一覧を取得。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM barcode_registry ORDER BY created_at DESC"
            ).fetchall()

    def delete_barcode(self, barcode: str):
        """バーコード登録を削除。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM barcode_registry WHERE barcode = ?", (barcode,)
            )

    # ── 撮影記録 ──
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
    """カメラ映像取得・動体検知・バーコード読み上げを行うスレッド。

    トリガー (動体検知時に即発火):
        IDLE ──(背景変化あり)──▶ TRIGGERED → ダイアログ表示 → 撮影 → クールダウン → IDLE

    バーコード検知は撮影トリガーとは独立。
    検知時はTTSで薬品名（登録済みなら薬名）を読み上げるのみ。
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
        self.state = STATE_IDLE
        self.motion_area_ratio = 0.0   # デバッグ表示用

        # ── 撮影制御 ──
        self.is_capturing = False
        self.capture_progress = 0
        self.cooldown_until = 0.0      # 撮影後の再検知抑制

        # ── バーコード読み上げ（撮影とは独立） ──
        self.last_spoken_barcode: str | None = None
        self.barcode_speak_cooldown = 0.0
        self.last_detected_barcode: dict | None = None  # 直近の検知結果

    # ──────────────────────────────────
    #  メインループ
    # ──────────────────────────────────
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.app.after(0, lambda: self.app.set_status("エラー: カメラを開けませんでした"))
            return

        # 最大解像度を要求（カメラが対応する最大値が適用される）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)

        # 実際に適用された解像度を取得
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = max(int(actual_fps), 1)
        self.ring_buffer = deque(maxlen=max(RING_BUFFER_SEC * self.fps, 10))

        self.running = True
        res_info = f"カメラ接続済み — {actual_w}x{actual_h} @ {self.fps}fps"
        res_label = f"解像度: {actual_w} x {actual_h}  |  {self.fps} fps"
        self.app.after(0, lambda: self.app.set_status(res_info))
        self.app.after(0, lambda: self.app.camera_res_var.set(res_label))
        self.app.after(100, lambda: self.app._sync_camera_sliders())
        print(res_info)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            timestamp = datetime.now()
            self.ring_buffer.append((frame.copy(), timestamp))

            # ── 計数モード ──
            if self.app.is_counting_mode():
                display, count, shapes = self._process_pill_counting(frame)
                self.app.after(0, lambda d=display: self.app.update_counting_preview(d))
                self.app.after(0, lambda c=count, s=shapes: self.app._update_pill_count(c, s))
                time.sleep(1.0 / self.fps)
                continue

            display = frame.copy()

            if not self.is_capturing and not self.app.is_barcode_register_mode():
                self._step_state_machine(frame, display)

            # バーコード読み上げ（撮影トリガーとは独立）
            self._scan_barcode_for_tts(frame, display)

            self._draw_overlay(display)
            self.app.after(0, lambda f=display: self.app.update_preview(f))

            time.sleep(1.0 / self.fps)

        if self.cap:
            self.cap.release()

    # ──────────────────────────────────
    #  動体検知トリガー
    # ──────────────────────────────────
    def _step_state_machine(self, frame, display):
        has_motion = self._detect_motion(frame)

        # ── 撮影後クールダウン中 ──
        if time.time() < self.cooldown_until:
            self.state = STATE_IDLE
            return

        # ── IDLE: 変化待ち → 動体検知で即撮影開始 ──
        if self.state == STATE_IDLE:
            if has_motion:
                self.state = STATE_TRIGGERED
                self.is_capturing = True
                self.capture_progress = 0
                self.app.after(0, lambda: self.app._start_capture())

        # ── TRIGGERED: 撮影中 ──
        elif self.state == STATE_TRIGGERED:
            pass

    def _detect_motion(self, frame):
        """背景差分 (MOG2) で動体を判定。

        Returns:
            has_motion: 背景と比較して有意な変化領域がある
        """
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        fg_pixels = np.count_nonzero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        self.motion_area_ratio = fg_pixels / total_pixels
        return self.motion_area_ratio > MOTION_AREA_RATIO

    # ──────────────────────────────────
    #  バーコード読み上げ（撮影トリガーとは独立）
    # ──────────────────────────────────
    BARCODE_TTS_COOLDOWN = 10.0  # 同一バーコードの再読み上げ抑制(秒)

    def _scan_barcode_for_tts(self, frame, display):
        """バーコードを検出し、見つかったらTTSで読み上げる（撮影とは無関係）。
        登録済みなら薬名を、未登録ならバーコードデータを読み上げる。
        """
        now = time.time()
        if now < self.barcode_speak_cooldown:
            # クールダウン中でも検知結果は描画する
            if self.last_detected_barcode:
                self._draw_barcode(display, self.last_detected_barcode)
            return

        barcodes = decode_barcodes(frame)
        if not barcodes:
            if self.last_detected_barcode is not None:
                self.last_detected_barcode = None
                self.app.after(0, lambda: self.app.medicine_name_var.set(""))
            return

        bc = barcodes[0]
        self.last_detected_barcode = bc
        self._draw_barcode(display, bc)

        # バーコード登録タブの表示を更新
        self.app.after(0, lambda d=bc["data"]: self.app._update_barcode_scan_display(d))

        # 同じバーコードは連続で読み上げない
        if bc["data"] != self.last_spoken_barcode:
            self.last_spoken_barcode = bc["data"]
            self.barcode_speak_cooldown = now + self.BARCODE_TTS_COOLDOWN

            # 登録済みなら薬名を読み上げ、未登録ならバーコードデータ
            medicine_name = None
            if self.app.db:
                medicine_name = self.app.db.lookup_barcode(bc["data"])

            if medicine_name:
                self.app.tts.speak(medicine_name)
                print(f"薬名読み上げ: {medicine_name} ({bc['type']} — {bc['data']})")
            else:
                self.app.tts.speak(f"バーコード検知。{bc['data']}")
                print(f"バーコード読み上げ: {bc['type']} — {bc['data']}")

    def _draw_barcode(self, display, bc):
        """バーコード枠とラベルを描画。登録済みなら薬名も表示。"""
        if bc["polygon"]:
            pts = np.array(bc["polygon"], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
        elif bc["rect"]:
            x, y, w, h = bc["rect"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # 登録済み薬名があればラベルに表示
        medicine_name = None
        if self.app.db:
            medicine_name = self.app.db.lookup_barcode(bc["data"])

        if medicine_name:
            label = f'{medicine_name} ({bc["type"]})'
            color = (0, 255, 255)  # 黄色で登録済みを示す
        else:
            label = f'{bc["type"]}: {bc["data"][:40]}'
            color = (0, 255, 0)

        cv2.putText(
            display, label, (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

    # ──────────────────────────────────
    #  計数モード処理
    # ──────────────────────────────────
    def _process_pill_counting(self, frame):
        """黒背景上の錠剤を計数し、輪郭・番号を描画したフレームを返す。
        距離変換＋ウォーターシェッドで接触した錠剤を分離する。
        """
        display = frame.copy()
        threshold = self.app.pill_threshold_var.get()

        # 1. グレースケール → ぼかし → 二値化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (PILL_BLUR_KSIZE, PILL_BLUR_KSIZE), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # 2. モルフォロジー（ノイズ除去）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. 距離変換＋ウォーターシェッドで接触錠剤を分離
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # 確実な背景（膨張で前景の外側）
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # マーカー生成
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1           # 背景を1に
        markers[unknown == 255] = 0     # 未知領域を0に

        # ウォーターシェッド実行
        ws_input = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.watershed(ws_input, markers)

        # 4. 各ラベルの輪郭を取得して分類
        shape_counts = {"丸": 0, "楕円": 0, "カプセル": 0}
        pill_num = 0
        num_labels = markers.max()

        for label_id in range(2, num_labels + 1):  # 0=未知, 1=背景, 2~=前景
            mask = np.uint8(markers == label_id) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)

            area = cv2.contourArea(cnt)
            if area < PILL_MIN_AREA or area > PILL_MAX_AREA:
                continue

            pill_num += 1
            perimeter = cv2.arcLength(cnt, True)

            # 形状分類
            shape = "楕円"
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > PILL_CIRCULARITY_ROUND:
                    shape = "丸"

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (ma, MA), _ = ellipse
                aspect_ratio = max(ma, MA) / (min(ma, MA) + 1e-6)
                if aspect_ratio > PILL_ASPECT_CAPSULE:
                    shape = "カプセル"

            shape_counts[shape] += 1

            # 描画: 輪郭（色分け）
            color = {
                "丸": (0, 255, 0),
                "楕円": (255, 200, 0),
                "カプセル": (0, 200, 255),
            }[shape]
            cv2.drawContours(display, [cnt], -1, color, 2)

            # 番号を重心に描画
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    display, str(pill_num), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )

        # 合計をオーバーレイ
        cv2.putText(
            display, f"COUNT: {pill_num}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3,
        )

        return display, pill_num, shape_counts

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
                STATE_TRIGGERED: ((0, 255, 0), "TRIGGERED"),
            }
            color, state_text = state_colors.get(self.state, ((255, 255, 255), self.state))

        cv2.putText(display, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # デバッグ情報
        if not self.is_capturing:
            info = f"BG:{self.motion_area_ratio:.4f}"
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
        self.cooldown_until = time.time() + MOTION_COOLDOWN_SEC
        self.state = STATE_IDLE


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

        saved_config = self._load_config()
        self.base_dir = ctk.StringVar(value=saved_config.get("base_dir", DEFAULT_BASE_DIR))
        self.camera_index = ctk.IntVar(value=saved_config.get("camera_index", 0))

        self.camera_thread: CameraThread | None = None
        self.tts = TTSWorker()
        self.tts.start()
        self.db: Database | None = None
        self._photo_ref = None  # PhotoImage の参照保持（GC防止）
        self._counting_photo_ref = None  # 計数プレビュー用
        self.camera_res_var = ctk.StringVar(value="解像度: ---")

        self._build_ui()
        self._init_db()

    def _load_config(self) -> dict:
        """設定ファイルを読み込む。"""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_config(self):
        """現在の設定をファイルに保存する。"""
        config = {
            "base_dir": self.base_dir.get(),
            "camera_index": self.camera_index.get(),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"設定保存エラー: {e}")

    def _init_db(self):
        os.makedirs(self.base_dir.get(), exist_ok=True)
        self.db = Database(self.base_dir.get())
        # バーコード登録一覧を更新
        if hasattr(self, "reg_list_frame"):
            self._refresh_barcode_list()

    # ────────────────────────────────────
    #  UI 構築
    # ────────────────────────────────────
    def _build_ui(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        self._build_camera_tab()
        self._build_barcode_tab()
        self._build_counting_tab()
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

        self.btn_test = ctk.CTkButton(
            ctrl, text="テスト撮影", width=100,
            fg_color="#e67e22", hover_color="#d35400",
            command=self.test_capture, state="disabled",
        )
        self.btn_test.pack(side="left", padx=5)

        # 2段目: 保存先
        ctrl2 = ctk.CTkFrame(tab)
        ctrl2.pack(fill="x", padx=5, pady=(0, 5))

        ctk.CTkLabel(ctrl2, text="保存先:").pack(side="left", padx=(10, 2))
        ctk.CTkEntry(ctrl2, textvariable=self.base_dir, width=400).pack(side="left", padx=2)
        ctk.CTkButton(ctrl2, text="参照", width=50, command=self._browse_dir).pack(
            side="left", padx=5
        )
        ctk.CTkButton(ctrl2, text="フォルダを開く", width=100, command=self._open_save_dir).pack(
            side="left", padx=5
        )

        # 3段目: 解像度表示 ＋ 明るさ・コントラスト・露出
        ctrl3 = ctk.CTkFrame(tab)
        ctrl3.pack(fill="x", padx=5, pady=(0, 5))

        ctk.CTkLabel(
            ctrl3, textvariable=self.camera_res_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00ccff",
        ).pack(side="left", padx=(10, 20))

        # 明るさ
        ctk.CTkLabel(ctrl3, text="明るさ:").pack(side="left", padx=(5, 2))
        self.brightness_var = ctk.DoubleVar(value=0)
        self.brightness_slider = ctk.CTkSlider(
            ctrl3, from_=-64, to=64, variable=self.brightness_var,
            width=120, command=lambda v: self._apply_camera_prop(cv2.CAP_PROP_BRIGHTNESS, v),
        )
        self.brightness_slider.pack(side="left", padx=2)

        # コントラスト
        ctk.CTkLabel(ctrl3, text="コントラスト:").pack(side="left", padx=(10, 2))
        self.contrast_var = ctk.DoubleVar(value=0)
        self.contrast_slider = ctk.CTkSlider(
            ctrl3, from_=0, to=100, variable=self.contrast_var,
            width=120, command=lambda v: self._apply_camera_prop(cv2.CAP_PROP_CONTRAST, v),
        )
        self.contrast_slider.pack(side="left", padx=2)

        # 露出
        ctk.CTkLabel(ctrl3, text="露出:").pack(side="left", padx=(10, 2))
        self.exposure_var = ctk.DoubleVar(value=0)
        self.exposure_slider = ctk.CTkSlider(
            ctrl3, from_=-10, to=0, variable=self.exposure_var,
            width=120, command=lambda v: self._apply_camera_prop(cv2.CAP_PROP_EXPOSURE, v),
        )
        self.exposure_slider.pack(side="left", padx=2)

        # シャープネス
        ctk.CTkLabel(ctrl3, text="シャープネス:").pack(side="left", padx=(10, 2))
        self.sharpness_var = ctk.DoubleVar(value=0)
        self.sharpness_slider = ctk.CTkSlider(
            ctrl3, from_=0, to=255, variable=self.sharpness_var,
            width=120, command=lambda v: self._apply_camera_prop(cv2.CAP_PROP_SHARPNESS, v),
        )
        self.sharpness_slider.pack(side="left", padx=2)

        # プレビュー（tk.Label で高速な映像更新）
        self.preview = tk.Label(tab, bg="#1a1a1a", text="カメラ未接続", fg="gray")
        self.preview.pack(fill="both", expand=True, padx=5, pady=5)

        # 薬品名表示ラベル
        self.medicine_name_var = ctk.StringVar(value="")
        self.medicine_name_label = ctk.CTkLabel(
            tab, textvariable=self.medicine_name_var,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#00e0ff",
            height=50,
        )
        self.medicine_name_label.pack(fill="x", padx=5, pady=(0, 5))

    def _build_barcode_tab(self):
        """バーコード登録タブを構築。"""
        tab = self.tabview.add("バーコード登録")

        # ── 上部: 検知バーコード表示 ──
        scan_frame = ctk.CTkFrame(tab)
        scan_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            scan_frame, text="バーコードをカメラに見せてください",
            font=ctk.CTkFont(size=14),
        ).pack(pady=(10, 5))

        self.reg_barcode_var = ctk.StringVar(value="（スキャン待ち...）")
        self.reg_barcode_label = ctk.CTkLabel(
            scan_frame, textvariable=self.reg_barcode_var,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3498db",
        )
        self.reg_barcode_label.pack(pady=5)

        # ── 中央: 薬名入力 + 登録ボタン ──
        input_frame = ctk.CTkFrame(tab)
        input_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(input_frame, text="薬名:").pack(side="left", padx=(10, 5))
        self.reg_name_entry = ctk.CTkEntry(input_frame, width=300, placeholder_text="薬の名前を入力")
        self.reg_name_entry.pack(side="left", padx=5)

        self.btn_register = ctk.CTkButton(
            input_frame, text="登録", width=100,
            fg_color="#27ae60", hover_color="#219a52",
            command=self._register_barcode,
        )
        self.btn_register.pack(side="left", padx=10)

        # ── 下部: 登録済み一覧 ──
        ctk.CTkLabel(
            tab, text="登録済みバーコード一覧",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).pack(fill="x", padx=10, pady=(10, 2))

        self.reg_list_frame = ctk.CTkScrollableFrame(tab)
        self.reg_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 内部状態: 登録タブで検知したバーコード
        self._reg_scanned_barcode: str | None = None

    def _register_barcode(self):
        """バーコードと薬名をDBに登録する。"""
        barcode = self._reg_scanned_barcode
        name = self.reg_name_entry.get().strip()

        if not barcode:
            self.set_status("エラー: バーコードをスキャンしてください")
            return
        if not name:
            self.set_status("エラー: 薬名を入力してください")
            return

        self.db.register_barcode(barcode, name)
        self.tts.speak(f"{name}を登録しました")
        self.set_status(f"登録完了: {barcode} → {name}")
        self.reg_name_entry.delete(0, "end")
        self._reg_scanned_barcode = None
        self.reg_barcode_var.set("（スキャン待ち...）")
        self._refresh_barcode_list()

    def _refresh_barcode_list(self):
        """登録済みバーコード一覧を更新する。"""
        for w in self.reg_list_frame.winfo_children():
            w.destroy()

        if not self.db:
            return

        rows = self.db.get_all_registered()
        if not rows:
            ctk.CTkLabel(
                self.reg_list_frame, text="登録データなし", text_color="gray"
            ).pack(pady=20)
            return

        for row in rows:
            rf = ctk.CTkFrame(self.reg_list_frame)
            rf.pack(fill="x", padx=5, pady=2)

            info = f"{row['barcode']}  →  {row['name']}"
            ctk.CTkLabel(rf, text=info, anchor="w").pack(
                side="left", padx=10, pady=5, fill="x", expand=True
            )

            bc = row["barcode"]
            ctk.CTkButton(
                rf, text="削除", width=50,
                fg_color="#e74c3c", hover_color="#c0392b",
                command=lambda b=bc: self._delete_barcode(b),
            ).pack(side="right", padx=5, pady=5)

    def _delete_barcode(self, barcode: str):
        """バーコード登録を削除する。"""
        self.db.delete_barcode(barcode)
        self.set_status(f"削除: {barcode}")
        self._refresh_barcode_list()

    def _update_barcode_scan_display(self, barcode_data: str):
        """バーコード検知時の表示更新（メインスレッドから呼ぶ）。"""
        self._reg_scanned_barcode = barcode_data
        # 登録済みか確認
        existing = self.db.lookup_barcode(barcode_data) if self.db else None

        # カメラタブの薬品名ラベルを更新
        if existing:
            self.medicine_name_var.set(existing)
            self.medicine_name_label.configure(text_color="#00e0ff")
        else:
            self.medicine_name_var.set(f"未登録: {barcode_data[:30]}")
            self.medicine_name_label.configure(text_color="#888888")

        # バーコード登録タブの表示も更新
        if existing:
            self.reg_barcode_var.set(f"{barcode_data} （登録済み: {existing}）")
        else:
            self.reg_barcode_var.set(barcode_data)

    def _build_counting_tab(self):
        """計数モードタブを構築。"""
        tab = self.tabview.add("計数")

        # ── 上部: しきい値スライダー ──
        ctrl = ctk.CTkFrame(tab)
        ctrl.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(ctrl, text="しきい値:").pack(side="left", padx=(10, 5))
        self.pill_threshold_var = ctk.IntVar(value=PILL_THRESH_DEFAULT)
        self.pill_threshold_slider = ctk.CTkSlider(
            ctrl, from_=10, to=200, number_of_steps=190,
            variable=self.pill_threshold_var, width=250,
        )
        self.pill_threshold_slider.pack(side="left", padx=5)
        ctk.CTkLabel(ctrl, textvariable=self.pill_threshold_var, width=40).pack(
            side="left", padx=2
        )
        ctk.CTkButton(
            ctrl, text="リセット", width=80,
            command=lambda: self.pill_threshold_var.set(PILL_THRESH_DEFAULT),
        ).pack(side="left", padx=10)

        # ── 中央: プレビュー ──
        self.counting_preview = tk.Label(tab, bg="#000000", text="カメラ未接続", fg="gray")
        self.counting_preview.pack(fill="both", expand=True, padx=5, pady=5)

        # ── 下部: 計数結果 ──
        result_frame = ctk.CTkFrame(tab)
        result_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.pill_count_var = ctk.StringVar(value="合計: 0 個")
        ctk.CTkLabel(
            result_frame, textvariable=self.pill_count_var,
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="#00ff88",
        ).pack(side="left", padx=20, pady=10)

        self.pill_shape_var = ctk.StringVar(value="丸: 0  |  楕円: 0  |  カプセル: 0")
        ctk.CTkLabel(
            result_frame, textvariable=self.pill_shape_var,
            font=ctk.CTkFont(size=18),
            text_color="#aaaaaa",
        ).pack(side="left", padx=20, pady=10)

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
        self.btn_test.configure(state="normal")
        self.set_status("カメラ起動中...")

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_test.configure(state="disabled")
        self.preview.configure(image="", text="カメラ未接続")
        self._photo_ref = None
        self.set_status("停止")

    def is_barcode_register_mode(self) -> bool:
        """現在バーコード登録タブが選択されているか。"""
        try:
            return self.tabview.get() == "バーコード登録"
        except Exception:
            return False

    def is_counting_mode(self) -> bool:
        """現在計数タブが選択されているか。"""
        try:
            return self.tabview.get() == "計数"
        except Exception:
            return False

    def _apply_camera_prop(self, prop_id, value):
        """スライダー操作時にカメラプロパティを変更する。"""
        if self.camera_thread and self.camera_thread.cap and self.camera_thread.cap.isOpened():
            self.camera_thread.cap.set(prop_id, float(value))

    def _sync_camera_sliders(self):
        """カメラから現在のプロパティ値を読み取ってスライダーに反映する。"""
        if not self.camera_thread or not self.camera_thread.cap:
            return
        cap = self.camera_thread.cap
        try:
            b = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            c = cap.get(cv2.CAP_PROP_CONTRAST)
            e = cap.get(cv2.CAP_PROP_EXPOSURE)
            s = cap.get(cv2.CAP_PROP_SHARPNESS)
            self.brightness_var.set(b)
            self.contrast_var.set(c)
            self.exposure_var.set(e)
            self.sharpness_var.set(s)
        except Exception:
            pass

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

    def update_counting_preview(self, frame):
        """計数モードのプレビュー更新。"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.counting_preview.configure(image=photo, text="")
            self._counting_photo_ref = photo
        except Exception:
            pass

    def _update_pill_count(self, total: int, shapes: dict):
        """計数結果をUIラベルに反映。"""
        self.pill_count_var.set(f"合計: {total} 個")
        self.pill_shape_var.set(
            f"丸: {shapes['丸']}  |  楕円: {shapes['楕円']}  |  カプセル: {shapes['カプセル']}"
        )

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.base_dir.get())
        if d:
            self.base_dir.set(d)
            self._save_config()
            self._init_db()

    def _open_save_dir(self):
        """保存先フォルダを OS のファイルマネージャで開く。"""
        d = self.base_dir.get()
        os.makedirs(d, exist_ok=True)
        open_file(d)

    def test_capture(self):
        """テスト撮影: バーコード検知なしで現在のフレームを1枚保存する。"""
        if not self.camera_thread or not self.camera_thread.ring_buffer:
            self.set_status("エラー: カメラが起動していません")
            return

        self._init_db()
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_dir.get(), date_dir)
        os.makedirs(save_dir, exist_ok=True)

        frame, ts = self.camera_thread.get_current_frame()
        if frame is None:
            self.set_status("エラー: フレームを取得できませんでした")
            return

        ts = ts or now
        filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_test.jpg"
        filepath = os.path.join(save_dir, filename)

        if save_frame_as_jpg(filepath, frame):
            self.db.insert(
                barcode="TEST",
                barcode_type="TEST",
                filename=filename,
                filepath=filepath,
                captured_at=ts.strftime("%Y-%m-%d %H:%M:%S"),
                session_id=f"test_{ts.strftime('%Y%m%d_%H%M%S')}",
            )
            self.set_status(f"テスト撮影完了: {filepath}")
            self.tts.speak("テスト撮影完了")
        else:
            self.set_status(f"エラー: テスト撮影の保存に失敗しました")

    # ────────────────────────────────────
    #  撮影シーケンス
    # ────────────────────────────────────
    def _start_capture(self):
        """動体検知から直接呼ばれる。即座に撮影を開始する。"""
        self.tts.speak("撮影開始")
        threading.Thread(
            target=self._capture_sequence,
            daemon=True,
        ).start()

    def _capture_sequence(self):
        try:
            self._do_capture()
        except Exception as e:
            print(f"撮影シーケンスでエラー: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self.set_status(f"エラー: {e}"))
        finally:
            if self.camera_thread:
                self.camera_thread.is_capturing = False
                self.camera_thread.capture_progress = 0
                self.camera_thread.reset_for_next()

    def _do_capture(self):
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        date_dir = now.strftime("%Y-%m-%d")
        save_dir = os.path.join(self.base_dir.get(), date_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 直近のバーコード検知結果があれば DB に記録用に取得
        barcode_data = "N/A"
        barcode_type = "N/A"
        if self.camera_thread and self.camera_thread.last_detected_barcode:
            bc = self.camera_thread.last_detected_barcode
            barcode_data = bc["data"]
            barcode_type = bc["type"]

        self.after(0, lambda: self.set_status(f"撮影中 0/{CAPTURE_COUNT}"))
        saved_count = 0

        for i in range(CAPTURE_COUNT):
            if i == 0:
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
                lambda c=count: self.set_status(f"撮影中 {c}/{CAPTURE_COUNT}"),
            )

        msg = f"撮影完了: {saved_count}/{CAPTURE_COUNT}枚保存"
        self.after(0, lambda: self.set_status(msg))
        print(msg)
        print(f"保存先: {save_dir}")
        self.tts.speak(f"登録完了。{saved_count}枚保存しました")

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
        self._save_config()
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
