# 🧠 AI Human Activity Classification Platform

A production-style Django + TensorFlow web application that recognises **non-sensitive
human activities** (standing, sitting, walking, running, jumping, lying down, raising hands)
in real time from a PC webcam or a mobile phone camera.

> ⚠️ **Privacy by design.** The system classifies *movement only*. It never infers
> gender, race, ethnicity, religion, health status or any other sensitive attribute,
> and it performs no face recognition.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 Live camera | WebRTC `getUserMedia()`, front/rear camera switching, auto-predict loop |
| 🖼️ Image upload | Drag & drop, validation, side-by-side preview and result |
| 🧠 Deep learning | Custom CNN, MobileNetV2, EfficientNetB0 (transfer learning) |
| 🔌 REST API | `/api/predict/`, `/api/upload/`, `/api/history/`, `/api/classes/`, `/api/model-status/`, `/api/stats/` |
| 📊 Dashboard | Stat cards + Chart.js graphs (frequency, time series, confidence) |
| 🗂️ History | Search, filter, paginate, delete |
| 🔈 Voice feedback | `window.speechSynthesis` with volume and rate control |
| 🗄️ Dataset manager | Create classes, upload images, launch training from the browser |
| 📈 Evaluation | Accuracy, precision, recall, F1, confusion matrix (`ml/evaluate.py`) |
| 🌗 UI | Bootstrap 5, responsive, dark/light mode, mobile navigation |
| 🔐 Security | CSRF, upload validation, size limits, env-based secrets, staff-only actions |

---

## 🏗️ Architecture

```
Camera (PC / phone)
      ↓  WebRTC getUserMedia()
Capture frame → canvas → JPEG data URL
      ↓  POST /api/predict/
Django REST Framework
      ↓
Preprocessing:  resize 224×224  →  x_normalized = x / 255
      ↓
CNN / MobileNetV2 / EfficientNetB0
      ↓
Softmax → activity + confidence
      ↓
Prediction model (SQLite / PostgreSQL)
      ↓
Dashboard · History · Voice feedback
```

---

## 🚀 Installation (Windows + PyCharm + Python 3.11)

```powershell
git clone <your-repo-url> human_activity_ai
cd human_activity_ai

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env
python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"
# paste the output into DJANGO_SECRET_KEY inside .env

python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 0.0.0.0:8000
```

Open **http://127.0.0.1:8000/**

### Access from your phone
1. Both devices on the same Wi-Fi.
2. `ipconfig` → note the IPv4 address (e.g. `192.168.1.10`).
3. On the phone: `http://192.168.1.10:8000/`
4. Allow Python through Windows Firewall (Private networks).

> On iOS Safari and Android Chrome, camera access requires a **secure context**.
> `localhost` counts as secure; a LAN IP does not. To use the phone camera, either
> serve over HTTPS or use Chrome's flag
> `chrome://flags/#unsafely-treat-insecure-origin-as-secure` and add
> `http://192.168.1.10:8000`.

---

## 🗂️ Dataset preparation

Create one folder per activity inside `ml/dataset/`:

```
ml/dataset/
├── standing/       (>= 100 images)
├── sitting/
├── walking/
├── running/
├── jumping/
├── lying/
└── raising_hands/
```

Guidelines:
- 100–500 images per class is a good starting point.
- Vary background, lighting, clothing and camera angle.
- Use **only images you have the right to use** (your own recordings, or datasets
  with a licence that permits your intended use). Do **not** scrape personal photos.

You can also upload images through the **Dataset** page in the web UI.

---

## 🏋️ Training

```powershell
python ml/training/train.py --model mobilenetv2 --epochs 15
python ml/training/train.py --model cnn --epochs 20
python ml/training/train.py --model efficientnet --epochs 20 --fine-tune
```

Pipeline:

```
Dataset → Preprocessing → Training → Validation → Evaluation → Save model → Django
```

Artefacts written to `ml/models/`:
- `activity_model.keras` — the trained network
- `labels.json` — class names + validation accuracy

Restart the Django server (or POST `/api/train/`) so the new model is loaded.

---

## 📊 Evaluation

```powershell
python ml/evaluate.py
```

Prints and saves accuracy, precision, recall, F1-score and the confusion matrix to
`ml/models/evaluation.json`.

---

## 🔌 API documentation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict/` | Base64 data URL **or** multipart image → prediction |
| `POST` | `/api/upload/` | Multipart image → prediction (stored with thumbnail) |
| `GET` | `/api/history/` | Paginated history (`page`, `page_size`, `search`, `activity`, `source`) |
| `DELETE` | `/api/history/?id=N` | Delete one row (`?all=true` = staff only) |
| `GET` | `/api/classes/` | Activity classes + image counts |
| `GET` | `/api/model-status/` | Model loaded?, classes, threshold, path |
| `GET` | `/api/stats/` | Aggregates for the dashboard |
| `GET/POST/DELETE` | `/api/dataset/` | Dataset browse / upload / delete |
| `GET/POST` | `/api/train/` | Training status / start a run |

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

```json
{
  "activity": "running",
  "confidence": 0.97,
  "confidence_percent": 97.0,
  "processing_time_ms": 42.1,
  "uncertain": false,
  "demo_mode": false,
  "model": "activity_model",
  "speech": "Running detected."
}
```

Low confidence is never forced into a wrong label:

```json
{ "activity": "uncertain", "confidence": 0.38, "uncertain": true }
```

---

## 🎥 Camera permissions

| Platform | Notes |
|---|---|
| Windows Chrome/Edge | Prompt appears once; allow it. Camera needs `localhost` or HTTPS. |
| Android Chrome | Allow camera. Use **Switch Camera** for front/rear. |
| iOS Safari | Allow camera. HTTPS or `localhost` required. |
| Firefox | Works on `localhost`/HTTPS. |

If permission is denied: click the 🔒 icon in the address bar → *Site settings* →
Camera → Allow → reload.

---

## 🧪 Tests

```powershell
python manage.py test
```

---

## ☁️ Cloud storage (optional)

Set `USE_S3=True` and the AWS variables in `.env`, then `pip install django-storages boto3`.
The app runs perfectly on local disk when these are unset.

---

## 🚢 Deployment checklist

1. `DJANGO_DEBUG=False`
2. Strong `DJANGO_SECRET_KEY` (never committed)
3. `DJANGO_ALLOWED_HOSTS=yourdomain.com`
4. PostgreSQL via `DATABASE_URL`
5. `python manage.py collectstatic`
6. Run behind Gunicorn + Nginx over HTTPS (HTTPS is required for mobile cameras)
7. `SECURE_SSL_REDIRECT=True`, `SESSION_COOKIE_SECURE=True`, `CSRF_COOKIE_SECURE=True`

```powershell
gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 3
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| Camera doesn't start | Use `localhost` or HTTPS; check browser permissions |
| `TensorFlow not found` | `pip install tensorflow==2.16.1` inside the venv |
| Model not loading | Check `ml/models/activity_model.keras` exists; see `/api/model-status/` |
| "demo mode" banner | No trained model yet — train one |
| Phone can't reach the server | Same Wi-Fi? `runserver 0.0.0.0:8000`? Firewall allowed? |
| Slow prediction | Use MobileNetV2, reduce capture width in `camera.js` |
| Poor accuracy | Add more varied images; use `--fine-tune`; check class balance |

---

## 📄 License

MIT — use it, learn from it, build on it. Dataset images keep their own licences.   