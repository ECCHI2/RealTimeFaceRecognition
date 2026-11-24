# FaceVision — Gerçek Zamanlı Yüz Tespiti

**Bilgisayar Görmesi | Vize Projesi**  
Hazırlayan: MOHAMMAD ALSAYEH

## Projenin amacı

FaceVision; Haar Cascade, MediaPipe (BlazeFace) ve YOLOv8n-Face yöntemleriyle resim, video ve kamera görüntülerinde yüz tespiti gerçekleştiren bir bilgisayarlı görü uygulamasıdır. Projenin vize aşamasında üç farklı yaklaşımın uygulanması, CPU/GPU performansının gözlemlenmesi ve tek bir kullanıcı arayüzünden yönetilmesi hedeflenmiştir. Kimlik tanıma (face recognition) bu aşamanın kapsamı dışındadır.

## Özellikler

| Model | Resim | Kamera | Video | Cihaz |
|---|---|---|---|---|
| Haar Cascade | ✓ | ✓ | ✓ | CPU |
| MediaPipe | ✓ | ✓ | ✓ | CPU |
| YOLOv8n-Face | ✓ | ✓ | ✓ | Resim/kamera: GPU; video: CPU veya GPU |

- Tkinter tabanlı Türkçe kullanıcı arayüzü; model, kaynak ve cihaz seçimi.
- Python ortamının otomatik seçimi; çalışma günlüğü ve mevcut ölçümlerin gösterimi.
- OpenCV penceresinde yüz sınırlayıcı kutuları; resim/video sonuçlarının yerel kaydı.
- Dosya seçici (v2.1; mevcut betiklerle Windows üzerinde son kullanıcı testi yapılmalıdır).
- Normal çıkış ve gerektiğinde süreçlerin zorla sonlandırılması.

## Proje yapısı

```text
RealTimeFaceRecognition/
├── src/
│   ├── check_opencv.py
│   ├── face_detection_image.py
│   ├── face_detection_camera.py
│   ├── face_detection_video.py
│   ├── mediapipe_image.py
│   ├── mediapipe_camera.py
│   ├── mediapipe_video.py
│   ├── yolo_image.py
│   ├── yolo_camera.py
│   ├── yolo_video.py
│   ├── runtime_env.py
│   └── main_gui_final.py
├── data/                       # Yerel giriş dosyaları
├── models/                     # Yerel model ağırlıkları
├── results/                    # Yerel çıktı dosyaları
├── experiments/
│   └── benchmark_results.csv
├── requirements-cpu.txt
├── requirements-yolo.txt
├── run_facevision.bat
├── .gitignore
└── README.md
```

Yerel `.venv` ve `.venv_yolo` klasörleri Git deposuna dahil edilmez. OpenCV paketlerinin çakışmasını önlemek için iki ayrı ortam kullanılmaktadır.

## Kurulum (Windows)

Test edilen Python: **3.11.9**. Python'un Tkinter bileşeni ve GPU kullanımı için uyumlu NVIDIA sürücüsü gereklidir. Mevcut çalışma ortamları varsa yeniden kurulum gerekmez. Yeni bir bilgisayarda proje kökünde PowerShell üzerinden:

```powershell
py -3.11 -m venv .venv
py -3.11 -m venv .venv_yolo
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-cpu.txt
& ".\.venv_yolo\Scripts\python.exe" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& ".\.venv_yolo\Scripts\python.exe" -m pip install -r requirements-yolo.txt
& ".\.venv\Scripts\python.exe" -m pip check
& ".\.venv_yolo\Scripts\python.exe" -m pip check
```

Bu liste çalışır durumdaki ortamların ana paket sürümlerinden hazırlanmıştır; temiz kurulum ayrıca doğrulanmalıdır. GPU kontrolü:

```powershell
& ".\.venv_yolo\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"
```

Model dosyaları için [models/README.md](models/README.md) dosyasına bakınız. Bağımsız betiklerin varsayılan girdileri `data/test.jpg` ve `data/test_video.mp4` dosyalarıdır. Kendi izinli test dosyalarınızı kullanınız.

## Çalıştırma

1. `run_facevision.bat` dosyasına çift tıklayın veya PyCharm'da `src/main_gui_final.py` dosyasını çalıştırın.
2. Model ve giriş kaynağı seçin. Resim/video için **Dosya seç…** düğmesini kullanın; kamera için dosya seçilmez.
3. Uygun cihazı seçin ve **Tespiti başlat** düğmesine basın.
4. Görüntü penceresinde `Q` ile çıkın veya arayüzde **Durdur** düğmesini kullanın. Tam video sonuçları için işlemin doğal olarak bitmesini bekleyin.
5. **Sonuç klasörünü aç** ile çıktı dosyalarını inceleyin. Mevcut betikler sabit çıktı adlarını kullandığından sonraki deney önceki çıktının üzerine yazabilir.

## Deneysel sonuçlar

Test sistemi: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB), PyTorch 2.9.1+cu128, CUDA runtime 12.8. Video: 1280 × 720, 855 kare, yaklaşık 29,62 kaynak FPS.

| Model | Cihaz | Ort. tespit süresi (ms) | Detection FPS |
|---|---|---:|---:|
| Haar Cascade | CPU | 16,97 | 58,92 |
| MediaPipe | CPU | 2,89 | 345,94 |
| YOLOv8n-Face | GPU | 32,13 | 31,13 |
| YOLOv8n-Face | CPU | 49,91 | 20,04 |

YOLOv8 GPU çalıştırmasında ortalama inference 23,47 ms; CPU çalıştırmasında 46,71 ms ölçülmüştür. Her iki çalıştırma da 855 karede toplam 1701 yüz tespiti üretmiştir. Bu sayı benzersiz kişi sayısı veya doğruluk oranı değildir. GPU aynı YOLO modeli için bu deneyde yaklaşık 1,55 kat daha yüksek detection FPS sağlamıştır.

Detection FPS, ölçülen tespit süresinden hesaplanır; kamera gösterim FPS'i değildir. Modellerin ölçüm sınırları henüz eşitlenmediğinden modeller arası hız tablosu ön karşılaştırmadır. Precision/Recall için etiketli veriyle doğruluk değerlendirmesi henüz yapılmamıştır.

## Sonraki aşama

Final aşamasında yüz tanıma, yüz öznitelikleri (embeddings) ve kimlik veri tabanı değerlendirilecektir. Vize raporu ve sunumu tamamlandığında depoya ayrıca eklenecektir.

## Kaynaklar

- [OpenCV](https://opencv.org/)
- [MediaPipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
- [Lindevs YOLOv8-Face](https://github.com/lindevs/yolov8-face)
- [Ultralytics](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/)

Model ve üçüncü taraf kütüphanelerin lisansları kendi kaynaklarında belirtilmiştir. Kişisel test görüntüleri, model ağırlıkları ve üretilen videolar Git deposuna dahil edilmemektedir.
