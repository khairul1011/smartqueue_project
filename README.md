# SmartQueue AI API

REST API untuk prediksi waktu tunggu pasien rumah sakit menggunakan FastAPI dan model TensorFlow.

## Struktur Project

```text
smartqueue_project/
├── app/
│   ├── __init__.py
│   └── app.py
├── deployment/
│   └── model/
│       ├── best_model.keras
│       ├── feature_scaler.save
│       └── target_scaler.save
├── experiments/
│   └── model dan artefak hasil eksperimen
├── models/
│   └── model lama atau artefak referensi
├── notebooks/
│   └── notebook eksperimen dan training
├── runtime.txt
├── requirements.txt
└── README.md
```

## Menjalankan Lokal

1. Buat virtual environment.
2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan server:

```bash
uvicorn app.app:app --reload
```

4. Buka dokumentasi API di:

```text
http://127.0.0.1:8000/docs
```

## File Model

Aplikasi memuat artefak final dari `deployment/model/`:

- `best_model.keras`: model TensorFlow/Keras final.
- `feature_scaler.save`: scaler untuk fitur input.
- `target_scaler.save`: scaler untuk mengembalikan hasil prediksi ke satuan menit.

Folder `experiments/` digunakan untuk menyimpan hasil eksperimen dari notebook, sedangkan API production hanya memakai file di `deployment/model/`.

## Deployment Render

Runtime Python ditentukan di `runtime.txt`:

```text
python-3.10.13
```

Gunakan start command berikut di Render:

```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 120 app.app:app
```

Render akan menginstall dependency dari `requirements.txt`, termasuk `tensorflow-cpu` agar build lebih ringan untuk environment CPU.
