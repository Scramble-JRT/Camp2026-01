"""
Python 顔認識サンプル
    - OpenCV YuNet : 顔検出 / 顔の傾き (ロール角)
    - OpenCV SFace : 顔認識 (誰か)
    - MediaPipe Face Landmarker : 目のつぶり度合い (0〜100%)

必要なもの:
    pip install opencv-python numpy mediapipe

使い方 (画像はすべてUSBカメラで撮影します):
    python face_recognition_sample.py
    → Tkinter の画面にカメラ映像・睡眠時間・警告レベル・状態が出ます
    → 目を閉じ続けると「睡眠」とみなし、10秒ごとに警告レベルが上がります

初回実行時に学習済みモデル (ONNX) を自動ダウンロードします。
"""

import math
import os
import subprocess
import sys
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import font as tkfont, ttk

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
KNOWN_DIR = BASE_DIR / "known_faces"

MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
    "landmarker": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}

COSINE_THRESHOLD = 0.363  # これ以上なら同一人物とみなす (SFace の推奨値)
TILT_THRESHOLD = 20       # 顔の傾きがこの角度(度)を超えたら「倒れている」
EYE_CLOSED_THRESHOLD = 40 # 目のつぶり度合いがこの値(%)以上なら「閉じている」(小さいほど閉じ判定が広い)


# 検出時に試す画像の向き: 回転なし / 時計回りに90度 / 反時計回りに90度 (横向きの顔用)
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]


def rotate_image(img, rot):
    return img if ROTATIONS[rot] is None else cv2.rotate(img, ROTATIONS[rot])


def rotate_points(shape, rot, points):
    """元画像の座標を、回転後の画像の座標に変換する"""
    h, w = shape[:2]
    if rot == 1:
        return [(h - 1 - y, x) for x, y in points]
    if rot == 2:
        return [(y, w - 1 - x) for x, y in points]
    return [(x, y) for x, y in points]


def rotate_points_back(shape, rot, points):
    """回転後の画像の座標を、元画像の座標に戻す"""
    h, w = shape[:2]
    if rot == 1:
        return [(y, h - 1 - x) for x, y in points]
    if rot == 2:
        return [(w - 1 - y, x) for x, y in points]
    return [(x, y) for x, y in points]


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def download_models():
    """モデルがなければダウンロードしてパスを返す"""
    MODEL_DIR.mkdir(exist_ok=True)
    paths = {}
    for key, (filename, url) in MODELS.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"ダウンロード中: {filename}")
            urllib.request.urlretrieve(url, path)
        paths[key] = str(path)
    return paths


class FaceRecognizer:
    def __init__(self):
        paths = download_models()
        self.detector = cv2.FaceDetectorYN.create(paths["detector"], "", (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(paths["recognizer"], "")
        self.known = {}  # 名前 -> 特徴ベクトル
        self.eyes_closed = False
        self.landmarker = self._create_landmarker(paths["landmarker"])

    @staticmethod
    def _create_landmarker(model_path):
        """MediaPipe Face Landmarker (目のつぶり度合い用) を作る"""
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=5,
            output_face_blendshapes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def tilt_angle(face):
        """両目を結ぶ線の角度から顔の傾き(度)を求める。
        0 = まっすぐ、+ = 画面の右に倒れている、- = 左に倒れている"""
        rx, ry, lx, ly = face[4:8]  # 画面左側の目, 画面右側の目
        return math.degrees(math.atan2(ly - ry, lx - rx))

    def eye_closure(self, img):
        """画像内の各顔について (鼻先の座標, 左目%, 右目%) のリストを返す。
        % は目をつぶっている度合い (0 = 全開, 100 = 完全に閉じている)
        横向き(90度)の顔も、画像を回転させて探す"""
        import mediapipe as mp

        eyes = []
        for rot in range(len(ROTATIONS)):
            rimg = rotate_image(img, rot)
            rh, rw = rimg.shape[:2]
            rgb = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.monotonic() * 1000)
            ts = self._last_ts = max(ts, getattr(self, "_last_ts", -1) + 1)  # 単調増加にする
            result = self.landmarker.detect_for_video(mp_image, ts)

            for landmarks, shapes in zip(result.face_landmarks, result.face_blendshapes):
                scores = {s.category_name: s.score for s in shapes}
                nose = (landmarks[1].x * rw, landmarks[1].y * rh)  # 1 = 鼻先
                nose = rotate_points_back(img.shape, rot, [nose])[0]
                eyes.append((nose,
                             scores.get("eyeBlinkLeft", 0) * 100,    # 本人の左目
                             scores.get("eyeBlinkRight", 0) * 100))  # 本人の右目
        return eyes

    def _detect_one(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return faces if faces is not None else np.empty((0, 15))

    def detect(self, img):
        """顔を検出。各行が [x, y, w, h, 目鼻口の座標..., スコア, 回転番号] の配列を返す。
        座標は元画像のもの。横向き(90度)の顔は画像を回転させて検出し、座標を元に戻す。
        回転番号は feature() で顔を正しい向きに直すために使う"""
        found = []
        for rot in range(len(ROTATIONS)):
            for f in self._detect_one(rotate_image(img, rot)):
                pts = rotate_points_back(
                    img.shape, rot, [(f[0], f[1]), (f[0] + f[2], f[1] + f[3])]
                    + [tuple(f[4 + i * 2: 6 + i * 2]) for i in range(5)])
                (x0, y0), (x1, y1) = pts[0], pts[1]
                row = np.empty(16, dtype=np.float32)
                row[:4] = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
                row[4:14] = np.array(pts[2:]).ravel()
                row[14], row[15] = f[14], rot
                found.append(row)

        # 同じ顔が複数の向きで検出された場合は、スコアが高い方だけ残す
        kept = []
        for row in sorted(found, key=lambda r: -r[14]):
            if all(_iou(row, k) < 0.3 for k in kept):
                kept.append(row)
        return np.array(kept) if kept else np.empty((0, 16))

    def feature(self, img, face):
        """顔を正規化して切り出し、128次元の特徴ベクトルを返す"""
        rot = int(face[15])
        rimg = rotate_image(img, rot)
        pts = rotate_points(img.shape, rot,
                            [(face[0], face[1]), (face[0] + face[2], face[1] + face[3])]
                            + [tuple(face[4 + i * 2: 6 + i * 2]) for i in range(5)])
        (x0, y0), (x1, y1) = pts[0], pts[1]
        rface = np.empty(15, dtype=np.float32)
        rface[:4] = min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        rface[4:14] = np.array(pts[2:]).ravel()
        rface[14] = face[14]
        aligned = self.recognizer.alignCrop(rimg, rface)
        return self.recognizer.feature(aligned)

    def register_folder(self, folder):
        """フォルダ内の画像を「ファイル名 = 名前」として登録"""
        folder.mkdir(exist_ok=True)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            faces = self.detect(img)
            if len(faces) == 0:
                print(f"顔が見つかりません: {path.name}")
                continue
            # 一番大きい顔を使う
            face = max(faces, key=lambda f: f[2] * f[3])
            self.known[path.stem] = self.feature(img, face)
            print(f"登録: {path.stem}")
        if not self.known:
            print(f"※ {folder} に顔写真がないため、検出のみ行います")

    def identify(self, feat):
        """登録済みの顔と比較し、(名前, スコア) を返す"""
        best_name, best_score = "unknown", 0.0
        for name, known_feat in self.known.items():
            score = self.recognizer.match(feat, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < COSINE_THRESHOLD:
            best_name = "unknown"
        return best_name, best_score

    def annotate(self, img):
        """画像内の顔を検出・認識し、枠・名前・傾き・目のつぶり度合いを描画する"""
        eyes = self.eye_closure(img)
        self.eyes_closed = False  # 目を閉じている顔が1つでもあれば True
        for face in self.detect(img):
            x, y, w, h = face[:4].astype(int)
            if self.known:
                name, score = self.identify(self.feature(img, face))
                label = f"{name} ({score:.2f})"
            else:
                name, label = "face", f"face ({face[14]:.2f})"
            color = (0, 200, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 顔の傾き
            tilt = self.tilt_angle(face)
            tilted = abs(tilt) > TILT_THRESHOLD
            tilt_text = f"Tilt {tilt:+.0f}deg" + (" TILTED" if tilted else "")
            cv2.putText(img, tilt_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255) if tilted else (255, 255, 255), 2)

            # 目のつぶり度合い (YuNet の鼻の位置に一番近い MediaPipe の顔を使う)
            nx, ny = face[8:10]
            near = [e for e in eyes if math.dist(e[0], (nx, ny)) < w * 0.5]
            if near:
                _, left, right = min(near, key=lambda e: math.dist(e[0], (nx, ny)))
                closed = (left + right) / 2 >= EYE_CLOSED_THRESHOLD
                self.eyes_closed = self.eyes_closed or closed
                eye_text = f"Eyes closed L{left:.0f}% R{right:.0f}%" + (" CLOSED" if closed else "")
                cv2.putText(img, eye_text, (x, y + h + 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 255) if closed else (255, 255, 255), 2)
            # 目・鼻・口の目印
            for i in range(5):
                px, py = face[4 + i * 2: 6 + i * 2].astype(int)
                cv2.circle(img, (px, py), 2, (255, 200, 0), -1)
        return img


CAMERA_ID = 1  # USBカメラ。PC内蔵カメラは 0。USBカメラが映らない場合は 2 なども試す


def open_camera(camera_id=CAMERA_ID):
    """USBカメラを開く (Windows では DirectShow を使うと起動が速い)"""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit("Webカメラを開けません (他のアプリが使用中でないか確認してください)")
    return cap


# --- 睡眠の判定と警告レベル -------------------------------------------------
SLEEP_CONFIRM_SEC = 2.0  # 目を閉じ続けてこの秒数たったら「睡眠」とみなす (まばたき対策)
WAKE_GRACE_SEC = 2.0     # 目が開いてもこの秒数以内なら睡眠を続けているとみなす
LEVEL_SEC = 10           # 睡眠が続いて次のレベルに上がるまでの秒数

# (画面に出す名前, ロボットの動作)
LEVELS = [
    ("警告", "警告音"),
    ("こちょこちょ", "こちょこちょ"),
    ("つんつん", "つんつん"),
    ("最終警告", "最終警告"),
    ("つねる", "つねる"),
    ("つねってひねる", "つねってひねる"),
]

# そのレベルになった瞬間に鳴らす音声 (キーは 0 始まり: 0 = Level1, 3 = Level4)
LEVEL_SOUNDS = {
    0: BASE_DIR / "起きてください.mp4",
    1: BASE_DIR / "LEVEL2こちょこちょ.mp4",
    2: BASE_DIR / "LEVEL3つんつん.mp4",
    3: BASE_DIR / "起きろ大石.mp4",
    4: BASE_DIR / "LEVEL5つねります.mp4",
    5: BASE_DIR / "LEVEL6つねりこちょこちょ.mp4",
}

# 起きたときに鳴らす音声。WAKE_MIN_LEVEL (0 始まり: 2 = Level3) 以上まで進んでいたときだけ
WAKE_SOUND = BASE_DIR / "おはようございます.mp4"
WAKE_MIN_LEVEL = 2

# Windows 標準の MediaPlayer で再生する (追加ライブラリ不要)。mp4 の音声だけを鳴らす
_PLAY_SCRIPT = (
    "Add-Type -AssemblyName presentationCore;"
    "$p = New-Object System.Windows.Media.MediaPlayer;"
    "$p.Open([Uri]$env:SOUND_PATH); $p.Play();"
    "for ($i = 0; $i -lt 100 -and -not $p.NaturalDuration.HasTimeSpan; $i++) { Start-Sleep -Milliseconds 100 };"
    "if ($p.NaturalDuration.HasTimeSpan) { Start-Sleep -Milliseconds ($p.NaturalDuration.TimeSpan.TotalMilliseconds + 300) }"
)


def play_sound(path):
    """音声を別プロセスで再生する (画面を止めない)。再生プロセスを返す"""
    if not path.exists():
        print(f"音声ファイルがありません: {path}")
        return None
    return subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", _PLAY_SCRIPT],
        env={**os.environ, "SOUND_PATH": str(path)},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


FONT = "Meiryo"
C_PANEL = "#c8d0da"
C_GREY_TEXT = "#5b6470"
C_RED = "#ff5757"
C_BLUE = "#5271ff"
C_PURPLE = "#5e1bf0"
C_GREEN = "#7ed957"


def rounded_points(x0, y0, x1, y1, r):
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    return [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
            x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]


def to_photo(img):
    """OpenCV 画像 (BGR) を Tk の PhotoImage にする。PPM 形式にすると PIL なしで渡せる"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return tk.PhotoImage(data=b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]) + rgb.tobytes())


def fit_image(img, max_w, max_h):
    """縦横比を保ったまま max_w x max_h に収まるように縮尺する"""
    h, w = img.shape[:2]
    k = min(max_w / w, max_h / h)
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)


class App:
    """Tkinter の画面: カメラ映像・睡眠時間・警告レベル・状態を表示する"""

    DW, DH = 1920, 1080  # 画面レイアウトを設計したサイズ。ウィンドウに合わせて拡大縮小する
    CAM_BOX = (38, 22, 1000, 712)  # カメラ映像のパネル
    TITLE_BOX = (1090, 30, 1640, 390)  # 「起きろ大石」の見出し画像を置く場所

    def __init__(self, root, fr):
        self.root, self.fr = root, fr
        self.cap = open_camera()
        self.raw = None     # 枠を描く前の最新フレーム
        self.result = None  # 枠を描いた最新フレーム
        self.sleep_start = None  # 目を閉じ始めた時刻
        self.last_closed = 0.0   # 最後に目を閉じていた時刻
        self.prev_level = -1     # 前のフレームのレベル (-1 = 睡眠していない)
        self.player = None       # 音声の再生プロセス
        root.title("Face Recognition")
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.s, self.ox, self.oy = 1.0, 0.0, 0.0  # 拡大率と、中央寄せのための余白
        self.sleep_sec = 0.0
        self.base_size = {}  # 文字の項目 -> 設計上の文字サイズ
        title = BASE_DIR / "title.png"
        self.title_img = cv2.imread(str(title)) if title.exists() else None
        self.canvas = tk.Canvas(root, width=self.DW, height=self.DH, bg="white",
                                highlightthickness=0)
        self.build_controls()  # 先にボタン行を下に置き、残りをキャンバスに使う
        self.canvas.pack(fill="both", expand=True)
        self.build_static()
        self.canvas.bind("<Configure>", self.on_resize)
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: self.toggle_fullscreen(False))
        self.toggle_fullscreen(True)
        self.update()

    # --- 画面の部品 ---------------------------------------------------------
    def sx(self, x):
        return self.ox + x * self.s

    def sy(self, y):
        return self.oy + y * self.s

    def scaled_box(self, box):
        x0, y0, x1, y1 = box
        return self.sx(x0), self.sy(y0), self.sx(x1), self.sy(y1)

    def panel(self, box, fill, r=30, tags=()):
        return self.canvas.create_polygon(
            rounded_points(*self.scaled_box(box), r * self.s), smooth=True,
            fill=fill, outline="", tags=tags)

    def text(self, x, y, s="", size=20, color="black", anchor="nw", bold=True):
        item = self.canvas.create_text(
            self.sx(x), self.sy(y), text=s, anchor=anchor, fill=color,
            font=self.font(size, bold))
        self.base_size[item] = size
        return item

    def font(self, size, bold=True):
        """画面の拡大率に合わせたフォント。マイナスの値はピクセル指定なので、
        Windows の表示スケール(125% など)に左右されず、レイアウトとずれない"""
        px = max(6, round(size * 4 / 3 * self.s))
        return tkfont.Font(family=FONT, size=-px, weight="bold" if bold else "normal")

    def fit_text(self, item, max_width):
        """文字が max_width (設計上の幅) をはみ出すときは、収まるまで小さくする"""
        text = self.canvas.itemcget(item, "text")
        size = self.base_size[item]
        f = self.font(size)
        while size > 8 and f.measure(text) > max_width * self.s:
            size -= 1
            f = self.font(size)
        self.canvas.itemconfigure(item, font=f)

    def on_resize(self, event):
        """ウィンドウの大きさに合わせてレイアウトを作り直す"""
        s = min(event.width / self.DW, event.height / self.DH)
        ox, oy = (event.width - self.DW * s) / 2, (event.height - self.DH * s) / 2
        if (s, ox, oy) == (self.s, self.ox, self.oy):
            return
        self.s, self.ox, self.oy = s, ox, oy
        self.canvas.delete("all")
        self.build_static()
        self.refresh_ui(self.sleep_sec)

    def toggle_fullscreen(self, on=None):
        if on is None:
            on = not self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", on)

    def build_static(self):
        c = self.canvas
        self.panel(self.CAM_BOX, "#d9d9d9", 40)
        self.cam_image = c.create_image(self.sx((self.CAM_BOX[0] + self.CAM_BOX[2]) / 2),
                                        self.sy((self.CAM_BOX[1] + self.CAM_BOX[3]) / 2))

        # 見出し (title.png があれば画像、なければ文字)
        tx0, ty0, tx1, ty1 = self.TITLE_BOX
        if self.title_img is not None:
            self.title_photo = to_photo(fit_image(
                self.title_img, (tx1 - tx0) * self.s, (ty1 - ty0) * self.s))
            c.create_image(self.sx((tx0 + tx1) / 2), self.sy((ty0 + ty1) / 2),
                           image=self.title_photo)
        else:
            self.text((tx0 + tx1) / 2, (ty0 + ty1) / 2, "起きろ!!\n大石!!", size=80,
                      anchor="center")

        self.level_pill = self.panel((1038, 475, 1468, 583), C_RED, 54)
        self.level_text = self.text(1070, 529, size=34, color=C_BLUE, anchor="w")
        self.text(1492, 462, "Next...", size=26)
        self.panel((1532, 518, 1910, 580), "#e3e8ee", 31)
        self.next_text = self.text(1545, 549, size=22, color=C_GREY_TEXT, anchor="w")

        self.panel((1070, 626, 1850, 726), C_PANEL, 50)
        self.panel((1088, 640, 1835, 712), C_PURPLE, 36)
        self.remain_text = self.text(1372, 778, size=30, anchor="center")

        self.sleep_time = self.text(1455, 975, size=38, anchor="w")

        self.panel((38, 735, 516, 986), C_PANEL, 50)
        self.text(77, 762, "状態", size=28)
        self.state_sleep = self.text(92, 900, "睡眠", size=60, anchor="w")
        self.state_normal = self.text(355, 897, "通常", size=28, anchor="w")

        self.panel((535, 752, 1083, 1053), C_PANEL, 50)
        self.level_rows = []
        for i, (name, action) in enumerate(LEVELS):
            y = 780 + i * 44.6
            self.text(588, y, f"Level{i + 1}", size=19, color=C_GREY_TEXT, anchor="w")
            self.level_rows.append(
                self.text(725, y, action, size=19, color=C_GREY_TEXT, anchor="w"))

    def build_controls(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="bottom", fill="x")
        self.status = tk.StringVar()
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=12)
        ttk.Button(bar, text="終了", command=self.quit).pack(side="right")
        ttk.Button(bar, text="全画面 (F11)", command=self.toggle_fullscreen).pack(side="right", padx=4)
        ttk.Button(bar, text="画面を保存", command=self.save).pack(side="right", padx=4)

    # --- 毎フレームの処理 ---------------------------------------------------
    def update(self):
        ok, frame = self.cap.read()
        if ok:
            self.raw = frame.copy()
            self.result = self.fr.annotate(frame)
            self.show_frame(self.result)
            self.track_sleep()
        self.root.after(15, self.update)

    def show_frame(self, img):
        x0, y0, x1, y1 = self.scaled_box(self.CAM_BOX)
        img = fit_image(img, x1 - x0 - 20 * self.s, y1 - y0 - 20 * self.s)
        self.photo = to_photo(img)
        self.canvas.itemconfigure(self.cam_image, image=self.photo)

    def track_sleep(self):
        """目を閉じている時間から睡眠時間を数え、画面を更新する"""
        now = time.monotonic()
        if self.fr.eyes_closed:
            self.last_closed = now
            if self.sleep_start is None:
                self.sleep_start = now
        elif self.sleep_start is not None and now - self.last_closed > WAKE_GRACE_SEC:
            self.sleep_start = None  # 起きた

        sleep_sec = 0.0
        if self.sleep_start is not None and self.last_closed - self.sleep_start >= SLEEP_CONFIRM_SEC:
            sleep_sec = now - self.sleep_start
        self.sleep_sec = sleep_sec
        self.refresh_ui(sleep_sec)

    def refresh_ui(self, sleep_sec):
        c = self.canvas
        sleeping = sleep_sec > 0
        c.itemconfigure(self.sleep_time, text=f"睡眠時間　{int(sleep_sec)}秒")

        # 状態 (今の状態を大きく色付きで、もう一方は小さく灰色で出す)
        c.itemconfigure(self.state_sleep, text="睡眠" if sleeping else "通常",
                        fill=C_RED if sleeping else "#2e9e4f")
        c.itemconfigure(self.state_normal, text="通常" if sleeping else "睡眠",
                        fill=C_GREY_TEXT)

        c.delete("progress")
        if not sleeping:
            level = -1
            c.itemconfigure(self.level_pill, fill="#e3e8ee")
            c.itemconfigure(self.level_text, text="待機中", fill=C_GREY_TEXT)
            c.itemconfigure(self.next_text, text=f"Level1　{LEVELS[0][0]}")
            c.itemconfigure(self.remain_text, text="")
        else:
            level = min(int(sleep_sec // LEVEL_SEC), len(LEVELS) - 1)
            last = level == len(LEVELS) - 1
            elapsed = LEVEL_SEC if last else sleep_sec - level * LEVEL_SEC
            c.itemconfigure(self.level_pill, fill=C_RED)
            c.itemconfigure(self.level_text, text=f"Level{level + 1}　{LEVELS[level][0]}",
                            fill=C_BLUE)
            c.itemconfigure(self.next_text,
                            text="なし" if last else f"Level{level + 2}　{LEVELS[level + 1][0]}")
            c.itemconfigure(self.remain_text,
                            text="最終レベル" if last else
                            f"残り{math.ceil(LEVEL_SEC - elapsed)}秒（{LEVEL_SEC}中）")
            if elapsed > 0:
                right = 1088 + (1835 - 1088) * elapsed / LEVEL_SEC
                c.create_polygon(rounded_points(*self.scaled_box((1088, 640, right, 712)),
                                                36 * self.s), smooth=True,
                                 fill=C_GREEN, outline="", tags="progress")

        # 対象のレベルに上がった瞬間に音声を鳴らす
        # (前の音声がまだ鳴っていれば止めて、新しいレベルの音声を優先する)
        if level != self.prev_level and level in LEVEL_SOUNDS:
            if self.player is not None and self.player.poll() is None:
                self.player.terminate()
            self.player = play_sound(LEVEL_SOUNDS[level])
        # Level3 以上まで進んでから起きたら「おはようございます」を鳴らす
        # (まだ警告音が鳴っていれば止めて、こちらを優先する)
        if level == -1 and self.prev_level >= WAKE_MIN_LEVEL:
            if self.player is not None and self.player.poll() is None:
                self.player.terminate()
            self.player = play_sound(WAKE_SOUND)
        self.prev_level = level

        # 帯の中に文字が収まるようにする
        self.fit_text(self.level_text, 370)
        self.fit_text(self.next_text, 345)

        # レベル一覧: 今のレベルだけ強調
        for i, item in enumerate(self.level_rows):
            c.itemconfigure(item, fill=C_RED if i == level else C_GREY_TEXT)

    # --- ボタン -------------------------------------------------------------
    def save(self):
        if self.result is None:
            return
        out = BASE_DIR / "capture_result.jpg"
        cv2.imwrite(str(out), self.result)
        self.status.set(f"保存しました: {out}")

    def quit(self):
        self.cap.release()
        self.root.destroy()


class Splash:
    """起動画面: UI1 (タイトル + PRESS ENTER) → UI2 (ローディング) → メイン画面 (UI3)"""

    DW, DH = 1920, 1080
    LOADING_MS = 3000      # ローディング画面を出す時間
    SPIN_MS = 110          # スピナーが1コマ回る間隔
    TITLE_CENTER = (970, 261)  # タイトル画像を置く中心 (title_big.png は UI1.png の切り出し)
    SPIN_CENTER, SPIN_RADIUS, DOT_RADIUS = (933, 679), 98, 22
    # 時計回りに進む8個の点。先頭ほど濃く、後ろに行くほど薄くする
    SPIN_SHADES = ["#333333", "#4d4d4d", "#666666", "#7d7d7d", "#929292", "#a8a8a8",
                   "#d0d0d0", "#f2f2f2"]

    def __init__(self, root, on_done):
        self.root, self.on_done = root, on_done
        self.phase = "title"  # title -> loading -> done
        self.s, self.ox, self.oy = 1.0, 0.0, 0.0
        self.spin_pos = 0
        self.after_ids = []
        img = BASE_DIR / "title_big.png"
        self.title_img = cv2.imread(str(img)) if img.exists() else None

        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        root.bind("<Return>", self.on_enter)
        root.bind("<KP_Enter>", self.on_enter)
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
        root.bind("<F11>", lambda e: root.attributes(
            "-fullscreen", not root.attributes("-fullscreen")))

    def sx(self, x):
        return self.ox + x * self.s

    def sy(self, y):
        return self.oy + y * self.s

    def on_resize(self, event):
        s = min(event.width / self.DW, event.height / self.DH)
        self.s = s
        self.ox, self.oy = (event.width - self.DW * s) / 2, (event.height - self.DH * s) / 2
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        cx, cy = self.TITLE_CENTER
        if self.title_img is not None:
            h, w = self.title_img.shape[:2]
            self.photo = to_photo(fit_image(self.title_img, w * self.s, h * self.s))
            c.create_image(self.sx(cx), self.sy(cy), image=self.photo)
        else:
            c.create_text(self.sx(cx), self.sy(cy), text="起きろ!!\n大石!!", fill="black",
                          font=tkfont.Font(family=FONT, size=-round(200 * self.s), weight="bold"))

        if self.phase == "title":
            c.create_text(self.sx(960), self.sy(601), text="[PRESS ENTER]", fill="black",
                          font=tkfont.Font(family=FONT, size=-round(54 * self.s), weight="bold"))
        else:
            self.draw_dots()

    def draw_dots(self):
        """スピナーの点だけを描き直す (タイトル画像の作り直しは重いので避ける)"""
        c = self.canvas
        c.delete("dots")
        sx0, sy0 = self.SPIN_CENTER
        for i in range(8):
            angle = math.radians(i * 45)  # i = 0 が真上で、時計回り
            x = sx0 + self.SPIN_RADIUS * math.sin(angle)
            y = sy0 - self.SPIN_RADIUS * math.cos(angle)
            r = self.DOT_RADIUS
            shade = self.SPIN_SHADES[(self.spin_pos - i) % 8]
            c.create_oval(self.sx(x - r), self.sy(y - r), self.sx(x + r), self.sy(y + r),
                          fill=shade, outline="", tags="dots")

    def on_enter(self, event=None):
        if self.phase != "title":
            return
        self.phase = "loading"
        self.draw()
        self.after_ids.append(self.root.after(self.SPIN_MS, self.spin))
        self.after_ids.append(self.root.after(self.LOADING_MS, self.finish))

    def spin(self):
        if self.phase != "loading":
            return
        self.spin_pos = (self.spin_pos + 1) % 8
        self.draw_dots()
        self.after_ids.append(self.root.after(self.SPIN_MS, self.spin))

    def finish(self):
        self.phase = "done"
        for i in self.after_ids:
            self.root.after_cancel(i)
        for key in ("<Return>", "<KP_Enter>", "<Escape>", "<F11>"):
            self.root.unbind(key)
        self.canvas.destroy()
        self.on_done()


if __name__ == "__main__":
    fr = FaceRecognizer()
    fr.register_folder(KNOWN_DIR)
    root = tk.Tk()
    root.title("Face Recognition")
    root.attributes("-fullscreen", True)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    Splash(root, lambda: App(root, fr))
    root.mainloop()
