from runtime_env import ensure_yolo_environment
ensure_yolo_environment()

import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO

# 1. Dosya yollari
project_path = Path(__file__).resolve().parent.parent

image_path = project_path / "data" / "test.jpg"

model_path = (
    project_path / "models" / "yolov8n-face-lindevs.pt"
)

result_path = project_path / "results" / "yolo_detected_image.jpg"

# 2. GPU kontrolu
if not torch.cuda.is_available():
    raise RuntimeError("CUDA kullanilamiyor!")

device = 0

print("GPU:", torch.cuda.get_device_name(0))

# 3. YOLO modelini yukle
model = YOLO(str(model_path))

print("YOLO modeli yuklendi.")

# 4. Resmi oku
image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Resim bulunamadi: {image_path}")

height, width = image.shape[:2]

print("Resim boyutu:", width, "x", height)

# 5. GPU warm-up
model.predict(
    image,
    imgsz=640,
    conf=0.5,
    device=device,
    verbose=False
)

torch.cuda.synchronize()

# 6. Yuz tespiti ve sure olcumu
start_time = time.perf_counter()

results = model.predict(
    image,
    imgsz=640,
    conf=0.5,
    device=device,
    verbose=False
)

torch.cuda.synchronize()

detection_time = time.perf_counter() - start_time

result = results[0]

# 7. Tespit edilen yuzler
faces = result.boxes

print("Bulunan yuz sayisi:", len(faces))

for box in faces:

    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    confidence = float(box.conf.item())

    print("Confidence:", round(confidence, 4))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Face: {confidence:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# 8. Performans bilgileri
detection_ms = detection_time * 1000

print("Toplam tespit suresi (ms):", round(detection_ms, 2))
print("Inference suresi (ms):", result.speed["inference"])
print("Cihaz:", "GPU")

# 9. Sonucu kaydet
result_path.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(result_path), image)

print("Sonuc kaydedildi:", result_path)

# 10. Ekran icin resmi kucult
scale = min(800 / width, 600 / height, 1.0)

small = cv2.resize(
    image,
    (int(width * scale), int(height * scale))
)

cv2.imshow("YOLOv8 Face Detection - GPU", small)

cv2.waitKey(0)
cv2.destroyAllWindows()
