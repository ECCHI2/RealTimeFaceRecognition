
import cv2
import mediapipe as mp
import time
from pathlib import Path

# 1. Dosya yollari
project_path = Path(__file__).resolve().parent.parent

video_path = project_path / "data" / "test_video.mp4"
model_path = project_path / "models" / "blaze_face_short_range.tflite"
result_path = project_path / "results" / "mediapipe_video.mp4"

# 2. MediaPipe modeli
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

# 3. Video dosyasini ac
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise RuntimeError("Video acilamadi!")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = video.get(cv2.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 30.0

# 4. Sonuc videosunu hazirla
result_path.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(result_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (width, height)
)

if not writer.isOpened():
    video.release()
    raise RuntimeError("Video kayit dosyasi acilamadi!")

frame_count = 0
total_detection_time = 0.0
last_timestamp = -1

# 5. Frame'leri isle
try:
    with FaceDetector.create_from_options(options) as detector:

        while True:

            ret, frame = video.read()

            if not ret:
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Frame timestamp
            timestamp_ms = max(
                int(frame_count * 1000 / video_fps),
                last_timestamp + 1
            )
            last_timestamp = timestamp_ms

            # Tespit suresini olc
            start_time = time.perf_counter()

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            detection_time = time.perf_counter() - start_time
            total_detection_time += detection_time

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
            detection_fps = (
                1.0 / detection_time
                if detection_time > 0 else 0
            )

            cv2.putText(
                frame,
                f"Faces: {len(result.detections)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Detection FPS: {detection_fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            writer.write(frame)
            frame_count += 1

            cv2.imshow("MediaPipe Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    video.release()
    writer.release()
    cv2.destroyAllWindows()

# 6. Ortalama performans
if frame_count > 0:
    average_ms = (
        total_detection_time / frame_count
    ) * 1000

    average_fps = (
        frame_count / total_detection_time
        if total_detection_time > 0 else 0
    )

    print("Islenen frame:", frame_count)
    print("Ortalama tespit suresi (ms):", round(average_ms, 2))
    print("Ortalama detection FPS:", round(average_fps, 2))
    print("Video kaydedildi:", result_path)
