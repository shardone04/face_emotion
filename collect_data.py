import os
import time
import csv
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------
# 감정 라벨 정의
# ----------------------------
EMOTIONS = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "anger",
    4: "fear",
    5: "surprise",
    6: "disgust",
}

# ----------------------------
# MediaPipe FaceMesh 랜드마크 인덱스
# ----------------------------
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

# 삼각형 정의 (논문 구조 기반)
TRIANGLES = [
    ("T1", "le_outer", "re_outer", "center"),
    ("T2", "le_inner", "re_inner", "center"),
    ("T3", "mouth_left", "mouth_right", "center"),
    ("T4", "mouth_left", "lip_upper", "lip_lower"),
    ("T5", "mouth_right", "lip_upper", "lip_lower"),
]


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def triangle_area(d1, d2, d3):
    s = (d1 + d2 + d3) / 2
    val = s * (s - d1) * (s - d2) * (s - d3)
    return np.sqrt(max(val, 0))


def compute_features(pA, pB, pC):
    d1 = euclidean(pA, pB)
    d2 = euclidean(pB, pC)
    d3 = euclidean(pC, pA)

    aot = triangle_area(d1, d2, d3)
    perimeter = d1 + d2 + d3
    r = (2 * aot / perimeter) if perimeter > 1e-6 else 0

    icc = 2 * np.pi * r
    icat = np.pi * (r ** 2)

    return aot, icc, icat


def main():
    os.makedirs("data", exist_ok=True)
    csv_path = "data/raw_features.csv"

    cap = cv2.VideoCapture(0)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    header = ["timestamp", "segment_id", "label_id", "label_name"]
    for tri, _, _, _ in TRIANGLES:
        header += [f"{tri}_AoT", f"{tri}_ICC", f"{tri}_ICAT"]

    file_exists = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(header)

    recording = False
    segment_id = 0
    current_label = 0

    print("Controls:")
    print("0~6 : 감정 선택")
    print("r : 녹화 시작/중지")
    print("q : 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]

            pts = {}
            for name, idx in LANDMARKS.items():
                lm = face_landmarks.landmark[idx]
                pts[name] = np.array([lm.x * w, lm.y * h])

            feature_row = []

            for tri_name, a, b, c in TRIANGLES:
                aot, icc, icat = compute_features(pts[a], pts[b], pts[c])
                feature_row += [aot, icc, icat]

                # 시각화
                cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 0, 0), 1)
                cv2.line(frame, tuple(pts[b].astype(int)), tuple(pts[c].astype(int)), (255, 0, 0), 1)
                cv2.line(frame, tuple(pts[c].astype(int)), tuple(pts[a].astype(int)), (255, 0, 0), 1)

            if recording:
                row = [time.time(), segment_id, current_label, EMOTIONS[current_label]] + feature_row
                writer.writerow(row)

        text = f"Label: {current_label} ({EMOTIONS[current_label]}) | Recording: {recording}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("FER Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            if recording:
                segment_id += 1
                print("Recording started")
            else:
                print("Recording stopped")
        elif key in [ord(str(i)) for i in range(7)]:
            current_label = int(chr(key))
            print("Label changed:", EMOTIONS[current_label])

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()