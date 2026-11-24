
import cv2
from pathlib import Path

# 1. Dosya yollarini belirle
project_path = Path(__file__).resolve().parent.parent
image_path = project_path / "data" / "test.jpg"
result_path = project_path / "results" / "detected_faces.jpg"

# 2. Resmi oku
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Resim bulunamadi!")

# 3. Gri seviyeye donustur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Hazir yuz tespit modelini yukle
model_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(model_path)

if face_detector.empty():
    raise RuntimeError("Yuz tespit modeli yuklenemedi!")

# 5. Yuzleri tespit et
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

print("Bulunan yuz sayisi:", len(faces))

# 6. Tespit edilen yuzleri isaretle
for (x, y, w, h) in faces:

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    cv2.putText(
        image,
        "Face",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# 7. Sonucu kaydet
result_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(result_path), image)

# 8. Ekrana uygun boyuta getir
height, width = image.shape[:2]

scale = min(800 / width, 650 / height, 1.0)

small = cv2.resize(
    image,
    (int(width * scale), int(height * scale))
)

# 9. Sonucu goster
cv2.imshow("Face Detection", small)

cv2.waitKey(0)
cv2.destroyAllWindows()
