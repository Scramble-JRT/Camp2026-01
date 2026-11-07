"""
Python 顔認識サンプル
    - OpenCV YuNet : 顔検出 / 顔の傾き (ロール角)
    - OpenCV SFace : 顔認識 (誰か)
    - MediaPipe Face Landmarker : 目のつぶり度合い (0〜100%)

必要なもの:
    pip install opencv-python numpy mediapipe

使い方 (画像はすべてPC内蔵のWebカメラで撮影します):
    1. 顔を登録 (known_faces/名前.jpg として保存されます)
        python face_recognition_sample.py register taro
       → カメラ映像に緑枠が出たら スペースキー で撮影、q で中止
       ※ 名前は英数字にしてください (OpenCV の文字描画は日本語非対応)
    2. 認識
        python face_recognition_sample.py
       → Webカメラ映像で認識 (s で画面を保存、q で終了)

初回実行時に学習済みモデル (ONNX) を自動ダウンロードします。
"""

import math
import sys
import time
import urllib.request
from pathlib import Path

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
EYE_CLOSED_THRESHOLD = 50 # 目のつぶり度合いがこの値(%)以上なら「閉じている」


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


def run_register(fr, name):
    """Webカメラで顔を撮影して known_faces/名前.jpg に保存する"""
    KNOWN_DIR.mkdir(exist_ok=True)
    cap = open_camera()
    print(f"「{name}」を登録します: 顔が1つ緑枠で映ったら スペースキー、q で中止")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        faces = fr.detect(frame)
        ready = len(faces) == 1
        color = (0, 200, 0) if ready else (0, 0, 255)
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        msg = "SPACE: save" if ready else "Show exactly one face"
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Register Face", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("中止しました")
            break
        if key == ord(" ") and ready:
            path = KNOWN_DIR / f"{name}.jpg"
            cv2.imwrite(str(path), frame)  # 枠なしの元画像を保存
            print(f"保存しました: {path}")
            break
    cap.release()
    cv2.destroyAllWindows()


def run_camera(fr):
    """Webカメラ映像でリアルタイムに顔認識する"""
    cap = open_camera()
    print("s キーで画面を保存、q キーで終了")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = fr.annotate(frame)
        cv2.imshow("Face Recognition", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = BASE_DIR / "capture_result.jpg"
            cv2.imwrite(str(out), result)
            print(f"保存しました: {out}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognizer()
    if len(sys.argv) >= 3 and sys.argv[1] == "register":
        run_register(fr, sys.argv[2])
    else:
        fr.register_folder(KNOWN_DIR)
        run_camera(fr)
