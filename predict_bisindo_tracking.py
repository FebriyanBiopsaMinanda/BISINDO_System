import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import urllib.request
import time

from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "resnet50_bisindo.h5"
LABEL_PATH = BASE_DIR / "class_names_bisindo.npy"
HAND_MODEL = BASE_DIR / "hand_landmarker.task"

IMG_SIZE = 224
CAMERA_INDEX = 0
PREDICT_INTERVAL = 5

# Fokus mendeteksi area bawah gambar
# 0.0 = seluruh gambar
# 0.25 = mulai dari 25% tinggi gambar
ROI_TOP = 0.20


print("Memuat model BISINDO...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

class_names = np.load(LABEL_PATH)

print("Model berhasil dimuat!")
print("Jumlah kelas:", len(class_names))

url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/"
    "1/hand_landmarker.task"
)

if not HAND_MODEL.exists():
    print("Mengunduh model hand tracking...")
    urllib.request.urlretrieve(url, HAND_MODEL)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30,
    min_tracking_confidence=0.30
)

# Koneksi landmark jari tangan
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]


def predict_bisindo(frame):

    img = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img, (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(
        img, verbose=0
    )[0]

    pred_idx = np.argmax(predictions)

    pred_class = str(class_names[pred_idx])
    confidence = float(predictions[pred_idx])

    return pred_class, confidence


def draw_hand_tracking(frame, hand_landmarks, y_offset, roi_height):

    height, width = frame.shape[:2]

    points = []

    for lm in hand_landmarks:

        x = int(lm.x * width)
        y = int(y_offset + lm.y * roi_height)

        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        points.append((int(x), int(y)))

    # Bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    padding = 20

    x_min = max(0, min(xs) - padding)
    x_max = min(width - 1, max(xs) + padding)

    y_min = max(0, min(ys) - padding)
    y_max = min(height - 1, max(ys) + padding)

    # Kotak hijau
    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2
    )

    # Garis landmark kuning
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 255),
            2
        )

    # Titik landmark merah
    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1
        )

    cv2.putText(
        frame,
        "HAND TRACKING",
        (x_min, max(25, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError(
        "Webcam tidak dapat dibuka. Periksa izin kamera."
    )

print("Webcam aktif!")
print("Tekan Q untuk keluar")
print("Tekan S untuk menyimpan gambar")

frame_count = 0
last_timestamp = -1

pred_class = None
confidence = 0.0

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Gagal membaca frame webcam")
                break

            # Simpan frame asli untuk prediksi
            original_frame = frame.copy()

            height, width = frame.shape[:2]

            # Fokus pada bagian gambar tempat tangan
            y_offset = int(height * ROI_TOP)
            roi = frame[y_offset:height, :]

            roi_height = roi.shape[0]

            # Memperbesar area deteksi
            roi = cv2.resize(
                roi, None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_LINEAR
            )

            rgb = cv2.cvtColor(
                roi, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            # Timestamp harus terus meningkat
            timestamp_ms = max(
                last_timestamp + 1,
                int(time.monotonic() * 1000)
            )
            last_timestamp = timestamp_ms

            # Tracking MediaPipe
            results = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            hand_count = len(results.hand_landmarks)

            # Jika tangan ditemukan
            if hand_count > 0:

                for landmarks in results.hand_landmarks:
                    draw_hand_tracking(
                        frame,
                        landmarks,
                        y_offset,
                        roi_height
                    )

                # Prediksi tiap beberapa frame
                if frame_count % PREDICT_INTERVAL == 0:
                    pred_class, confidence = predict_bisindo(
                        original_frame
                    )

            else:
                pred_class = None
                confidence = 0.0

            # Panel informasi
            cv2.rectangle(
                frame,
                (0, 0),
                (width, 90),
                (25, 25, 25),
                -1
            )

            if hand_count > 0 and pred_class is not None:
                text_pred = f"Prediksi BISINDO: {pred_class}"
                text_conf = f"Confidence: {confidence:.2%}"

                cv2.putText(
                    frame, text_pred,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame, text_conf,
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "Posisikan tangan di depan kamera",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            frame_count += 1

            # Menampilkan webcam
            cv2.imshow(
                "BISINDO Real-Time Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                filename = str(
                    BASE_DIR / "hasil_webcam_bisindo.jpg"
                )
                cv2.imwrite(filename, frame)
                print("Gambar disimpan:", filename)

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam ditutup")
