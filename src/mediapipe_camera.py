
import cv2
import mediapipe as mp
import time
from pathlib import Path

# 1. Model yolu
project_path = Path(__file__).resolve().parent.parent
model_path = project_path / "models" / "blaze_face_short_range.tflite"

# 2. MediaPipe ayarlari
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.VIDEO,
    min_detection_confidence=0.5
)

# 3. Kamerayi ac
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Kamera acilamadi!")

start = time.perf_counter()
previous_time = start
last_timestamp = -1
fps = 0.0

# 4. Modeli bir kez yukle
try:
    with FaceDetector.create_from_options(options) as detector:

        while True:
            ret, frame = camera.read()

            if not ret:
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe Image olustur
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Frame zamanini hesapla
            timestamp_ms = max(
                int((time.perf_counter() - start) * 1000),
                last_timestamp + 1
            )
            last_timestamp = timestamp_ms

            # Yuzleri tespit et
            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            # Yuzleri ciz
            for detection in result.detections:
                box = detection.bounding_box

                x = box.origin_x
                y = box.origin_y
                w = box.width
                h = box.height

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            # FPS hesapla
            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = 0.9 * fps + 0.1 * current_fps

            previous_time = current_time

            # Bilgileri ekrana yaz
            cv2.putText(
                frame,
                f"Faces: {len(result.detections)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.imshow("MediaPipe Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
