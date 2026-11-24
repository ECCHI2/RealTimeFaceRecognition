
from runtime_env import ensure_yolo_environment
ensure_yolo_environment()

import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO

# 1. Dosya yollari
project_path = Path(__file__).resolve().parent.parent

model_path = project_path / "models" / "yolov8n-face-lindevs.pt"
video_path = project_path / "data" / "test_video.mp4"

# 2. Cihaz secimi
print("1 - GPU")
print("2 - CPU")

choice = input("Cihaz seciniz [1/2]: ").strip()

if choice == "2":
    device = "cpu"
    device_name = "CPU"
else:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA kullanilamiyor!")

    device = 0
    device_name = "GPU"

print("Cihaz:", device_name)

result_path = (
    project_path / "results" /
    f"yolo_video_{device_name.lower()}.mp4"
)

result_path.parent.mkdir(parents=True, exist_ok=True)

# 3. Modeli yukle
model = YOLO(str(model_path))

# 4. Videoyu ac
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    cap.release()
    raise RuntimeError("Video acilamadi!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if video_fps <= 0:
    video_fps = 25

print("Video boyutu:", width, "x", height)
print("Video FPS:", video_fps)
print("Video frame sayisi:", video_frames)

# 5. Video kayit ayarlari
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(result_path),
    fourcc,
    video_fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    writer.release()
    raise RuntimeError("VideoWriter acilamadi!")

total_frames = 0
total_detection_time = 0
total_inference_time = 0
total_faces = 0

try:
    # 6. Warm-up
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Video okunamadi!")

    for _ in range(5):
        model.predict(
            frame,
            imgsz=640,
            conf=0.5,
            device=device,
            verbose=False
        )

    if device_name == "GPU":
        torch.cuda.synchronize()

    # Videonun basina don
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Video analizi baslatildi.")
    print("Cikis icin Q veya ESC.")

    # 7. Video yuz tespiti
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.5,
            device=device,
            verbose=False
        )

        if device_name == "GPU":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        result = results[0]
        faces = result.boxes

        inference_ms = result.speed["inference"]

        total_detection_time += elapsed
        total_inference_time += inference_ms
        total_frames += 1
        total_faces += len(faces)

        # 8. Bounding box
        for box in faces:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf.item())

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Face: {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 9. Performans bilgileri
        detection_fps = total_frames / total_detection_time

        information = [
            "Model: YOLOv8n-Face",
            f"Device: {device_name}",
            f"Detection FPS: {detection_fps:.2f}",
            f"Detection: {elapsed * 1000:.2f} ms",
            f"Inference: {inference_ms:.2f} ms",
            f"Faces: {len(faces)}",
            f"Frame: {total_frames}/{video_frames}"
        ]

        for i, text in enumerate(information):

            cv2.putText(
                frame,
                text,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # 10. Sonucu kaydet
        writer.write(frame)

        scale = min(900 / width, 650 / height, 1.0)

        preview = cv2.resize(
            frame,
            (max(1, int(width * scale)),
             max(1, int(height * scale)))
        )

        cv2.imshow(
            f"YOLOv8 Video - {device_name}",
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 11. Deney sonuclari
    if total_frames > 0:

        avg_ms = (
            total_detection_time / total_frames * 1000
        )

        avg_fps = (
            total_frames / total_detection_time
        )

        avg_inference = (
            total_inference_time / total_frames
        )

        print("\n========== DENEY SONUCLARI ==========")
        print("Model: YOLOv8n-Face")
        print("Cihaz:", device_name)
        print("Islenen frame:", total_frames)
        print("Toplam yuz tespiti:", total_faces)
        print(f"Ortalama tespit: {avg_ms:.2f} ms")
        print(f"Ortalama inference: {avg_inference:.2f} ms")
        print(f"Detection FPS: {avg_fps:.2f}")
        print("Sonuc:", result_path)
        print("=====================================")
