import base64
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications.resnet50 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50_bisindo.h5"
FINGER_MODEL_PATH = BASE_DIR / "finger_landmark_bisindo.h5"
FINGER_SCALER_PATH = BASE_DIR / "finger_landmark_scaler.npz"
LABEL_PATH = BASE_DIR / "class_names_bisindo.npy"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
GUIDE_IMAGE_PATH = (
    BASE_DIR / "bisindo-guide.jpg"
    if (BASE_DIR / "bisindo-guide.jpg").exists()
    else BASE_DIR / "bisindo guide.jpg"
)
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "Data"

IMG_SIZE = 224

# Indeks 3 tulang dari ujung tiap jari (tanpa titik 0 / Wrist):
FINGER_LANDMARK_INDICES = list(range(1, 21))
FINGER_CHAINS = [
    (4, 3, 2, 1),     # Ibu jari (3 tulang dari ujung)
    (8, 7, 6, 5),     # Telunjuk (3 tulang dari ujung)
    (12, 11, 10, 9),  # Jari tengah (3 tulang dari ujung)
    (16, 15, 14, 13), # Jari manis (3 tulang dari ujung)
    (20, 19, 18, 17), # Kelingking (3 tulang dari ujung)
]

app = FastAPI(
    title="BISINDO Sign-to-Text System",
    description="BISINDO Sign-to-Text Web Interface (3-Bone Finger Landmarks MLP + ResNet50)",
    version="3.0.0",
)

print("Memuat model BISINDO (ResNet50 & 3-Bone Finger Landmark MLP)...")
start_load = time.perf_counter()
resnet_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = [str(c) for c in np.load(LABEL_PATH)]

finger_model = None
scaler_mean = None
scaler_scale = None

if FINGER_MODEL_PATH.exists() and FINGER_SCALER_PATH.exists():
    finger_model = tf.keras.models.load_model(FINGER_MODEL_PATH, compile=False)
    scaler_data = np.load(FINGER_SCALER_PATH)
    scaler_mean = scaler_data["mean"]
    scaler_scale = scaler_data["scale"]
    # Warm-up finger model
    _dummy_feat = np.zeros((1, len(scaler_mean)), dtype=np.float32)
    finger_model(_dummy_feat, training=False)
    print("Model 3-Bone Finger Landmark berhasil dimuat!")

# Warm-up ResNet50
_dummy_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
resnet_model(_dummy_img, training=False)
load_time_ms = round((time.perf_counter() - start_load) * 1000, 2)
print(f"Semua model siap dalam {load_time_ms} ms!")


def extract_finger_features_single_hand_dict(landmarks: List[Dict[str, float]]) -> np.ndarray:
    pts = np.array([[lm["x"], lm["y"], lm.get("z", 0.0)] for lm in landmarks], dtype=np.float32)
    finger_pts = pts[FINGER_LANDMARK_INDICES]  # 20 titik (1..20, tanpa Wrist 0)

    mcp_center = np.mean(pts[[1, 5, 9, 13, 17]], axis=0)
    centered = finger_pts - mcp_center
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1)) + 1e-6
    norm_coords = (centered / scale).flatten()

    bone_features = []
    angle_features = []

    for tip, j1, j2, base in FINGER_CHAINS:
        b1 = (pts[tip] - pts[j1]) / scale
        b2 = (pts[j1] - pts[j2]) / scale
        b3 = (pts[j2] - pts[base]) / scale

        for b in (b1, b2, b3):
            length = float(np.linalg.norm(b))
            bone_features.extend([b[0], b[1], b[2], length])

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


def predict_from_finger_landmarks(hands_landmarks: List[List[Dict[str, float]]]) -> Dict[str, Any]:
    t0 = time.perf_counter()

    hands_sorted = sorted(
        hands_landmarks,
        key=lambda h: np.mean([h[idx]["x"] for idx in FINGER_LANDMARK_INDICES])
    )

    feat_h1 = extract_finger_features_single_hand_dict(hands_sorted[0])
    if len(hands_sorted) > 1:
        feat_h2 = extract_finger_features_single_hand_dict(hands_sorted[1])
        tips = [4, 8, 12, 16, 20]
        p1 = np.array([[hands_sorted[0][t]["x"], hands_sorted[0][t]["y"]] for t in tips], dtype=np.float32)
        p2 = np.array([[hands_sorted[1][t]["x"], hands_sorted[1][t]["y"]] for t in tips], dtype=np.float32)
        inter_tip_dists = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1).flatten()
        hand_count_flag = np.array([2.0], dtype=np.float32)
    else:
        feat_h2 = np.zeros_like(feat_h1)
        inter_tip_dists = np.zeros(25, dtype=np.float32)
        hand_count_flag = np.array([1.0], dtype=np.float32)

    raw_feat = np.concatenate([feat_h1, feat_h2, inter_tip_dists, hand_count_flag])
    scaled_feat = (raw_feat - scaler_mean) / (scaler_scale + 1e-8)
    batch = np.expand_dims(scaled_feat.astype(np.float32), axis=0)

    t_prep = time.perf_counter()
    preds = finger_model(batch, training=False).numpy()[0]
    t_infer = time.perf_counter()

    top_indices = np.argsort(preds)[::-1][:5]
    top_predictions = [
        {
            "label": class_names[int(idx)],
            "confidence": round(float(preds[int(idx)]), 4),
        }
        for idx in top_indices
    ]
    all_probs = {class_names[i]: round(float(preds[i]), 4) for i in range(len(class_names))}
    best = top_predictions[0]
    eps = 1e-12
    entropy = float(-np.sum(preds * np.log2(np.clip(preds, eps, 1.0))))

    return {
        "model_used": "finger_3bones_mlp",
        "prediction": best["label"],
        "confidence": best["confidence"],
        "top5": top_predictions,
        "all_probs": all_probs,
        "entropy": round(entropy, 4),
        "metrics": {
            "decode_ms": 0.0,
            "preprocess_ms": round((t_prep - t0) * 1000, 2),
            "inference_ms": round((t_infer - t_prep) * 1000, 2),
            "total_backend_ms": round((t_infer - t0) * 1000, 2),
        },
    }


def decode_and_predict_resnet(b64_image: str, return_preview: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_image)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise ValueError("Frame gambar tidak valid")

    t_decode = time.perf_counter()

    img_resized_bgr = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32)
    img_preprocessed = preprocess_input(img_float)
    batch = np.expand_dims(img_preprocessed, axis=0)

    t_preprocess = time.perf_counter()
    preds = resnet_model(batch, training=False).numpy()[0]
    t_infer = time.perf_counter()

    top_indices = np.argsort(preds)[::-1][:5]
    top_predictions = [
        {
            "label": class_names[int(idx)],
            "confidence": round(float(preds[int(idx)]), 4),
        }
        for idx in top_indices
    ]

    all_probs = {class_names[i]: round(float(preds[i]), 4) for i in range(len(class_names))}
    best = top_predictions[0]
    eps = 1e-12
    entropy = float(-np.sum(preds * np.log2(np.clip(preds, eps, 1.0))))

    res: Dict[str, Any] = {
        "model_used": "resnet50_image",
        "prediction": best["label"],
        "confidence": best["confidence"],
        "top5": top_predictions,
        "all_probs": all_probs,
        "entropy": round(entropy, 4),
        "input_resolution": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
        "metrics": {
            "decode_ms": round((t_decode - t0) * 1000, 2),
            "preprocess_ms": round((t_preprocess - t_decode) * 1000, 2),
            "inference_ms": round((t_infer - t_preprocess) * 1000, 2),
            "total_backend_ms": round((t_infer - t0) * 1000, 2),
        },
    }

    if return_preview:
        _, buf = cv2.imencode(".jpg", img_resized_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        res["preprocessed_preview"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return res


@app.get("/api/info")
async def get_model_info():
    devices = [d.name for d in tf.config.list_logical_devices()]
    return JSONResponse(
        {
            "model_name": "3-Bone Finger Landmark MLP + ResNet50",
            "finger_model_available": finger_model is not None,
            "input_shape": [IMG_SIZE, IMG_SIZE, 3],
            "num_classes": len(class_names),
            "classes": class_names,
            "tensorflow_version": tf.__version__,
            "devices": devices,
            "load_time_ms": load_time_ms,
            "hand_landmarker_available": HAND_MODEL_PATH.exists(),
            "dataset_available": DATA_DIR.exists(),
        }
    )


@app.get("/models/hand_landmarker.task")
async def get_hand_landmarker_model():
    return FileResponse(
        HAND_MODEL_PATH,
        media_type="application/octet-stream",
        filename="hand_landmarker.task",
    )


@app.get("/bisindo-guide.jpg")
async def get_bisindo_guide_image():
    return FileResponse(
        GUIDE_IMAGE_PATH,
        media_type="image/jpeg",
        filename="bisindo-guide.jpg",
    )


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_msg = await websocket.receive_text()
            payload = json.loads(raw_msg)
            model_type = payload.get("model_type", "finger_3bones")
            landmarks = payload.get("landmarks")
            frame_b64 = payload.get("image")
            client_ts = payload.get("client_ts", 0)
            return_preview = bool(payload.get("return_preview", False))

            try:
                if model_type == "finger_3bones" and finger_model is not None and landmarks:
                    result = predict_from_finger_landmarks(landmarks)
                elif frame_b64:
                    result = decode_and_predict_resnet(frame_b64, return_preview=return_preview)
                else:
                    await websocket.send_json({"error": "No valid input provided", "client_ts": client_ts})
                    continue

                result["client_ts"] = client_ts
                result["server_ts"] = int(time.time() * 1000)
                await websocket.send_json(result)
            except Exception as err:
                await websocket.send_json({"error": str(err), "client_ts": client_ts})
    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def simple_home_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/research")
async def research_dashboard_page():
    return FileResponse(STATIC_DIR / "research.html")
