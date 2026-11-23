import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

# 3 tulang dihitung dari ujung jari (sampai sendi pangkal jari / MCP):
# Setiap jari memiliki 4 titik untuk membentuk 3 segmen tulang dari ujung:
# - Ibu jari (Thumb):  TIP(4) -> IP(3)  -> MCP(2) -> CMC(1)   [3 tulang: (4,3), (3,2), (2,1)]
# - Telunjuk (Index):  TIP(8) -> DIP(7) -> PIP(6) -> MCP(5)   [3 tulang: (8,7), (7,6), (6,5)]
# - Tengah (Middle):   TIP(12)-> DIP(11)-> PIP(10)-> MCP(9)   [3 tulang: (12,11), (11,10), (10,9)]
# - Manis (Ring):      TIP(16)-> DIP(15)-> PIP(14)-> MCP(13)  [3 tulang: (16,15), (15,14), (14,13)]
# - Kelingking (Pinky):TIP(20)-> DIP(19)-> PIP(18)-> MCP(17)  [3 tulang: (20,19), (19,18), (18,17)]
# Catatan: Titik 0 (WRIST / Pergelangan & Telapak bawah) TIDAK dimasukkan sesuai permintaan user!

FINGER_LANDMARK_INDICES = list(range(1, 21))  # 20 titik (indeks 1..20, tanpa Wrist 0)

FINGER_CHAINS = [
    (4, 3, 2, 1),     # Ibu jari (3 tulang dari ujung)
    (8, 7, 6, 5),     # Telunjuk (3 tulang dari ujung)
    (12, 11, 10, 9),  # Jari tengah (3 tulang dari ujung)
    (16, 15, 14, 13), # Jari manis (3 tulang dari ujung)
    (20, 19, 18, 17), # Kelingking (3 tulang dari ujung)
]


def extract_finger_features_single_hand(landmarks) -> np.ndarray:
    """
    Mengekstrak fitur hanya dari 3 tulang tiap jari (indeks 1..20, tanpa Wrist 0):
    1. Koordinat relatif 20 titik jari (1..20) terhadap titik tengah pangkal jari (rata-rata MCP),
       dinormalisasi dengan skala ukuran jari agar tahan terhadap jarak kamera (60 fitur: 20 x 3).
    2. Vektor arah (dx, dy, dz) dan panjang dari 15 segmen tulang jari (3 tulang x 5 jari = 15 segmen -> 60 fitur).
    3. Sudut tekukan (cosine angle) antar tulang pada 2 sendi tiap jari (2 sudut x 5 jari = 10 fitur).
    Total per tangan = 130 fitur geometris tulang jari.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # shape (21, 3)

    # Gunakan hanya titik 1..20 (tanpa titik 0 / wrist)
    finger_pts = pts[FINGER_LANDMARK_INDICES]  # shape (20, 3)

    # Titik pusat referensi: rata-rata dari sendi pangkal ke-5 jari (indeks asli 1, 5, 9, 13, 17)
    mcp_center = np.mean(pts[[1, 5, 9, 13, 17]], axis=0)

    # Normalisasi translasi & skala
    centered = finger_pts - mcp_center
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1)) + 1e-6
    norm_coords = (centered / scale).flatten()  # 20 * 3 = 60 fitur

    bone_features = []
    angle_features = []

    for tip, j1, j2, base in FINGER_CHAINS:
        # 3 tulang dari ujung:
        # Tulang 1 (Distal): tip -> j1
        # Tulang 2 (Intermediate): j1 -> j2
        # Tulang 3 (Proximal / setelah sendi ke-2 dari ujung): j2 -> base
        b1 = (pts[tip] - pts[j1]) / scale
        b2 = (pts[j1] - pts[j2]) / scale
        b3 = (pts[j2] - pts[base]) / scale

        for b in (b1, b2, b3):
            length = float(np.linalg.norm(b))
            bone_features.extend([b[0], b[1], b[2], length])

        # Sudut pada sendi 1 (antara b1 dan b2) dan sendi 2 (antara b2 dan b3)
        for v_a, v_b in ((b1, b2), (b2, b3)):
            na = np.linalg.norm(v_a) + 1e-6
            nb = np.linalg.norm(v_b) + 1e-6
            cos_theta = float(np.clip(np.dot(v_a, v_b) / (na * nb), -1.0, 1.0))
            angle_features.append(cos_theta)

    return np.concatenate([
        norm_coords,
        np.array(bone_features, dtype=np.float32),
        np.array(angle_features, dtype=np.float32),
    ])


def extract_two_hand_finger_features(results) -> np.ndarray | None:
    """
    Menggabungkan fitur 3 tulang jari untuk maksimal 2 tangan (Slot Tangan Kiri & Slot Tangan Kanan,
    diurutkan berdasarkan posisi X dari kiri ke kanan gambar) + fitur jarak antar ujung jari kedua tangan.
    """
    if not results.hand_landmarks or len(results.hand_landmarks) == 0:
        return None

    # Urutkan tangan berdasarkan rata-rata X dari titik jari (1..20)
    hands_sorted = sorted(
        results.hand_landmarks,
        key=lambda h: np.mean([h[idx].x for idx in FINGER_LANDMARK_INDICES])
    )

    feat_h1 = extract_finger_features_single_hand(hands_sorted[0])  # 130 fitur
    if len(hands_sorted) > 1:
        feat_h2 = extract_finger_features_single_hand(hands_sorted[1])  # 130 fitur
        # Jarak relatif antara ke-5 ujung jari tangan 1 dan tangan 2 (penting untuk huruf 2 tangan BISINDO seperti A, B, P, M, N)
        tips = [4, 8, 12, 16, 20]
        p1 = np.array([[hands_sorted[0][t].x, hands_sorted[0][t].y] for t in tips], dtype=np.float32)
        p2 = np.array([[hands_sorted[1][t].x, hands_sorted[1][t].y] for t in tips], dtype=np.float32)
        inter_tip_dists = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1).flatten()  # 25 fitur
        hand_count_flag = np.array([2.0], dtype=np.float32)
    else:
        feat_h2 = np.zeros_like(feat_h1)
        inter_tip_dists = np.zeros(25, dtype=np.float32)
        hand_count_flag = np.array([1.0], dtype=np.float32)

    return np.concatenate([feat_h1, feat_h2, inter_tip_dists, hand_count_flag])  # Total: 130 + 130 + 25 + 1 = 286 fitur


def main():
    t_start = time.perf_counter()
    print("=== EKSTRAKSI LANDMARK 3 TULANG JARI (TANPA WRIST/TELAPAK) DARI DATASET BISINDO ===")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.20,
        min_hand_presence_confidence=0.20,
    )

    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    class_folders = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_folders]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    X_list = []
    y_list = []
    missed_per_class = Counter()
    detected_per_class = Counter()
    total_images = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for folder in class_folders:
            cname = folder.name
            cidx = class_to_idx[cname]
            files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in valid_ext]
            for fpath in files:
                total_images += 1
                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    missed_per_class[cname] += 1
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
                res = detector.detect(mp_img)

                feat = extract_two_hand_finger_features(res)
                if feat is None:
                    # Coba dengan sedikit padding jika tangan terlalu mepet ke tepi foto
                    padded = cv2.copyMakeBorder(img_rgb, 60, 60, 60, 60, cv2.BORDER_REPLICATE)
                    mp_img_pad = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(padded))
                    res = detector.detect(mp_img_pad)
                    feat = extract_two_hand_finger_features(res)

                if feat is not None:
                    X_list.append(feat)
                    y_list.append(cidx)
                    detected_per_class[cname] += 1
                else:
                    missed_per_class[cname] += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\nTotal gambar dipindai : {total_images}")
    print(f"Berhasil diekstrak    : {len(X)} ({len(X)/total_images:.2%})")
    print(f"Tidak terdeteksi MP   : {total_images - len(X)} ({(total_images - len(X))/total_images:.2%})")
    print(f"Dimensi Fitur Jari    : {X.shape[1]} fitur per sampel (20 titik jari/tangan, 3 tulang/jari, tanpa Wrist 0)")
    print("Deteksi per kelas     :", dict(detected_per_class))
    if missed_per_class:
        print("Gagal deteksi/kelas   :", dict(missed_per_class))

    # Strict Featurization Ordering: Split Train (70%), Val (15%), Test (15%) SEBELUM scaling!
    SEED = 42
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # Fit StandardScaler HANYA pada X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Augmentasi ringan pada data latih (flip/jitter koordinat kecil) untuk ketahanan webcam real-time
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 0.03, size=X_train_scaled.shape).astype(np.float32)
    X_train_aug = np.vstack([X_train_scaled, X_train_scaled + noise])
    y_train_aug = np.concatenate([y_train, y_train])

    # Bangun Model Deep Neural Network (MLP Residual / BatchNorm) untuk 3-Bone Finger Landmarks
    tf.random.set_seed(SEED)
    inputs = tf.keras.Input(shape=(X.shape[1],), name="finger_3bones_input")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="bisindo_class")(x)
    landmark_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Finger3Bones_Landmark_MLP")

    landmark_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0
        ),
    ]

    print("\n=== MELATIH MODEL LANDMARK 3 TULANG JARI ===")
    history = landmark_model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    best_val_acc = max(history.history["val_accuracy"])
    val_loss, val_acc = landmark_model.evaluate(X_val_scaled, y_val, verbose=0)
    test_loss, test_acc = landmark_model.evaluate(X_test_scaled, y_test, verbose=0)

    y_pred_probs = landmark_model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n=== HASIL EVALUASI MODEL LANDMARK 3 TULANG JARI ===")
    print(f"Validation Accuracy : {val_acc:.4%}")
    print(f"Test Accuracy       : {test_acc:.4%}")
    print(f"Test Macro F1-Score : {macro_f1:.4%}")
    print(f"Test Loss           : {test_loss:.4f}")
    print("\nClassification Report (Data Test):")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    # Simpan model dan parameter scaler agar bisa langsung dipakai di Web
    out_model_path = BASE_DIR / "finger_landmark_bisindo.h5"
    out_scaler_path = BASE_DIR / "finger_landmark_scaler.npz"
    landmark_model.save(out_model_path)
    np.savez(
        out_scaler_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        classes=np.array(class_names),
    )
    total_sec = time.perf_counter() - t_start
    print(f"Model disimpan ke   : {out_model_path.name} ({out_model_path.stat().st_size / 1024:.1f} KB)")
    print(f"Scaler disimpan ke  : {out_scaler_path.name}")
    print(f"Total waktu proses  : {total_sec:.1f} detik")


if __name__ == "__main__":
    main()
