from runtime_env import ensure_yolo_environment
ensure_yolo_environment()

import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO

# 1. Model yolu
project_path = Path(__file__).resolve().parent.parent
model_path = project_path / "models" / "yolov8n-face-lindevs.pt"

# 2. GPU kontrolu
if not torch.cuda.is_available():
    raise RuntimeError("CUDA kullanilamiyor!")

device = 0
print("GPU:", torch.cuda.get_device_name(0))

# 3. Modeli yukle
model = YOLO(str(model_path))
print("YOLOv8 Face modeli yuklendi.")

# 4. Kamerayi ac
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.release()
    raise RuntimeError("Kamera acilamadi!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

total_detection_time = 0
total_frames = 0
total_inference_time = 0

try:
    # 5. GPU warm-up
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Kameradan goruntu alinamadi!")

    for _ in range(5):
        model.predict(
            frame,
            imgsz=640,
            conf=0.5,
            device=device,
            verbose=False
        )

    torch.cuda.synchronize()

    print("Kamera baslatildi.")
    print("Cikis icin Q veya ESC tusuna basin.")

    # 6. Gercek zamanli yuz tespiti
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Tespit suresini olc
        start_time = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.5,
            device=device,
            verbose=False
        )

        torch.cuda.synchronize()

        detection_time = time.perf_counter() - start_time

        total_detection_time += detection_time
        total_frames += 1

        result = results[0]
        faces = result.boxes

        inference_ms = result.speed["inference"]
        total_inference_time += inference_ms

        # 7. Yuzleri ciz
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

        # 8. Performans hesaplama
        fps = total_frames / total_detection_time
        detection_ms = detection_time * 1000

        information = [
            "Model: YOLOv8n-Face",
            "Device: GPU",
            f"Detection FPS: {fps:.2f}",
            f"Detection: {detection_ms:.2f} ms",
            f"Inference: {inference_ms:.2f} ms",
            f"Faces: {len(faces)}"
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

        # 9. Goruntuyu goster
        cv2.imshow("YOLOv8 Face Detection - GPU", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    # 10. Kaynaklari serbest birak
    cap.release()
    cv2.destroyAllWindows()

    # 11. Deney sonuclari
    if total_frames > 0:

        avg_ms = total_detection_time / total_frames * 1000
        avg_fps = total_frames / total_detection_time
        avg_inference = total_inference_time / total_frames

        print("\n========== DENEY SONUCLARI ==========")
        print("Model: YOLOv8n-Face")
        print("Cihaz: GPU")
        print("Toplam frame:", total_frames)
        print(f"Ortalama tespit suresi: {avg_ms:.2f} ms")
        print(f"Ortalama inference: {avg_inference:.2f} ms")
        print(f"Detection FPS: {avg_fps:.2f}")
        print("=====================================")
