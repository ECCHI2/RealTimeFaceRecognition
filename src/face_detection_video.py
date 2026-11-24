
import cv2
import time
from pathlib import Path

# 1. Dosya yollari
project_path = Path(__file__).resolve().parent.parent

video_path = project_path / "data" / "test_video.mp4"
result_path = project_path / "results" / "detected_video.mp4"

# 2. Yuz tespit modelini yukle
model_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(model_path)

if face_detector.empty():
    raise RuntimeError("Yuz tespit modeli yuklenemedi!")

# 3. Video dosyasini ac
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise RuntimeError("Video acilamadi!")

# 4. Video bilgilerini al
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = video.get(cv2.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 25.0

print("Video Width:", width)
print("Video Height:", height)
print("Video FPS:", video_fps)

# 5. Sonuc videosunu hazirla
result_path.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(result_path),
    fourcc,
    video_fps,
    (width, height)
)

if not writer.isOpened():
    video.release()
    raise RuntimeError("Sonuc videosu olusturulamadi!")

# 6. Videoyu frame frame isle
frame_count = 0

total_detection_time = 0.0
try:
    while True:

        ret, frame = video.read()

        if not ret:
            break


        # Gri seviyeye donustur
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        # Sadece tespit suresini olc
        start_time = time.perf_counter()

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(80, 80)
        )

        detection_time = time.perf_counter() - start_time

        total_detection_time += detection_time

        # Yuzlerin etrafina kutu ciz
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Isleme hizini hesapla
        detection_fps = (
            1.0 / detection_time
            if detection_time > 0 else 0
        )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}  Detection FPS: {detection_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Islenmis frame'i kaydet
        writer.write(frame)
        frame_count += 1

        # Ekran icin kucult
        scale = min(800 / width, 600 / height, 1.0)

        small = cv2.resize(
            frame,
            (int(width * scale), int(height * scale))
        )

        cv2.imshow("Video Face Detection", small)

        # Q tusuna basinca cik
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    video.release()
    writer.release()
    cv2.destroyAllWindows()


if frame_count > 0 and total_detection_time > 0:

    average_ms = (
        total_detection_time / frame_count
    ) * 1000

    average_fps = (
        frame_count / total_detection_time
    )

    print("Islenen frame:", frame_count)
    print("Ortalama tespit suresi (ms):",
          round(average_ms, 2))
    print("Ortalama detection FPS:",
          round(average_fps, 2))


print("Islenen frame sayisi:", frame_count)
print("Sonuc kaydedildi:", result_path)
