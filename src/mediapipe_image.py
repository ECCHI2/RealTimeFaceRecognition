
import cv2
import mediapipe as mp
from pathlib import Path

# 1. Dosya yollari
project_path = Path(__file__).resolve().parent.parent

image_path = project_path / "data" / "test.jpg"

model_path = (
    project_path / "models" /
    "blaze_face_short_range.tflite"
)

result_path = (
    project_path / "results" /
    "mediapipe_detected.jpg"
)

# 2. Resmi oku
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Resim bulunamadi!")

if not model_path.is_file():
    raise FileNotFoundError("Model dosyasi bulunamadi!")

# 3. MediaPipe modelini hazirla
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.5
)

# 4. MediaPipe formatinda resmi oku
mp_image = mp.Image.create_from_file(str(image_path))

# 5. Yuzleri tespit et
with FaceDetector.create_from_options(options) as detector:
    result = detector.detect(mp_image)

print("Bulunan yuz sayisi:", len(result.detections))

# 6. Yuzlerin koordinatlarini al
for detection in result.detections:

    box = detection.bounding_box

    x = box.origin_x
    y = box.origin_y
    w = box.width
    h = box.height

    confidence = detection.categories[0].score

    print("Confidence:", confidence)

    # 7. Dikdortgen ciz
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Face: {confidence:.2f}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# 8. Sonucu kaydet
result_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(result_path), image)

# 9. Resmi kuculterek goster
height, width = image.shape[:2]

scale = min(800 / width, 650 / height, 1.0)

small = cv2.resize(
    image,
    (int(width * scale), int(height * scale))
)

cv2.imshow("MediaPipe Face Detection", small)

cv2.waitKey(0)
cv2.destroyAllWindows()
