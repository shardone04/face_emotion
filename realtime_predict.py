import argparse
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
import joblib

# Same mapping as collect_data.py
LANDMARKS = {
    "le_outer": 33,
    "le_inner": 133,
    "re_inner": 362,
    "re_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "lip_upper": 13,
    "lip_lower": 14,
    "center": 1,
}

EMOTIONS = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "anger",
    4: "fear",
    5: "surprise",
    6: "disgust",
}

TRIANGLES = [
    ("T1", "le_outer", "re_outer", "center"),
    ("T2", "le_inner", "re_inner", "center"),
    ("T3", "mouth_left", "mouth_right", "center"),
    ("T4", "mouth_left", "lip_upper", "lip_lower"),
    ("T5", "mouth_right", "lip_upper", "lip_lower"),
]


def euclid(p1, p2) -> float:
    return float(np.linalg.norm(p1 - p2))


def triangle_area_heron(d1, d2, d3) -> float:
    s = (d1 + d2 + d3) / 2.0
    val = s * (s - d1) * (s - d2) * (s - d3)
    return float(np.sqrt(max(val, 0.0)))


def triangle_features(pA, pB, pC):
    d1 = euclid(pA, pB)
    d2 = euclid(pB, pC)
    d3 = euclid(pC, pA)
    perim = d1 + d2 + d3
    aot = triangle_area_heron(d1, d2, d3)
    r = (2.0 * aot / perim) if perim > 1e-9 else 0.0
    icc = 2.0 * np.pi * r
    icat = np.pi * (r ** 2)
    return aot, icc, icat


def landmarks_to_points(face_landmarks, w, h):
    pts = {}
    for k, idx in LANDMARKS.items():
        lm = face_landmarks.landmark[idx]
        pts[k] = np.array([lm.x * w, lm.y * h], dtype=np.float32)
    return pts


def frame_feature_vector(pts):
    # returns dict of 15 base features
    out = {}
    for tname, a, b, c in TRIANGLES:
        aot, icc, icat = triangle_features(pts[a], pts[b], pts[c])
        out[f"{tname}_AoT"] = aot
        out[f"{tname}_ICC"] = icc
        out[f"{tname}_ICAT"] = icat
    return out


def window_stats(feature_frames, base_keys):
    """
    feature_frames: list of dicts (base features per frame)
    return stats dict for model input columns: *_mean, *_std, *_min, *_max
    """
    mat = np.array([[ff[k] for k in base_keys] for ff in feature_frames], dtype=np.float32)
    stats = {}
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    mn = mat.min(axis=0)
    mx = mat.max(axis=0)
    for i, k in enumerate(base_keys):
        stats[f"{k}_mean"] = float(mean[i])
        stats[f"{k}_std"] = float(std[i])
        stats[f"{k}_min"] = float(mn[i])
        stats[f"{k}_max"] = float(mx[i])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/rf_fer.joblib")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    payload = joblib.load(args.model)
    clf = payload["model"]
    X_cols = payload["feature_columns"]

    base_keys = []
    for tname, _, _, _ in TRIANGLES:
        base_keys += [f"{tname}_AoT", f"{tname}_ICC", f"{tname}_ICAT"]

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    buf = deque(maxlen=args.window)
    pred_label = "..."

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                pts = landmarks_to_points(res.multi_face_landmarks[0], w, h)
                feat = frame_feature_vector(pts)
                buf.append(feat)

                # Draw points/triangles (optional)
                for p in pts.values():
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

                # Predict when buffer full
                if len(buf) == args.window:
                    stats = window_stats(list(buf), base_keys)
                    X = np.array([[stats.get(c, 0.0) for c in X_cols]], dtype=np.float32)
                    yhat = int(clf.predict(X)[0])
                    pred_label = f"{yhat} ({EMOTIONS.get(yhat,'?')})"

            cv2.putText(frame, f"Pred: {pred_label}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Pred: {pred_label}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("FER Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()