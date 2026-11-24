import time
import cv2

# 1. Hazir yuz tespit modelini yukle
model_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(model_path)

if face_detector.empty():
    raise RuntimeError("Yuz tespit modeli yuklenemedi!")

# 2. Laptop kamerasini ac
camera = cv2.VideoCapture(0)

previous_time = time.perf_counter()
fps = 0.0

if not camera.isOpened():
    raise RuntimeError("Kamera acilamadi!")

# 3. Canli video dongusu
try:
    while True:

        # Kameradan bir frame oku
        ret, frame = camera.read()

        if not ret:
            print("Frame okunamadi!")
            break

        # Gri seviyeye donustur
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        # Yuzleri tespit et
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(80, 80)
        )

        # Yuzlerin etrafina dikdortgen ciz
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Bulunan yuz sayisini yaz
        cv2.putText(
            frame,
            "Faces: " + str(len(faces)),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
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

        # FPS degerini ekrana yaz
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Videoyu goster
        cv2.imshow("Live Face Detection", frame)

        # Q tusuna basinca cik
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Kamerayi serbest birak
    camera.release()
    cv2.destroyAllWindows()
