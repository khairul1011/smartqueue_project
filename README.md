# SmartQueue AI API

REST API untuk prediksi waktu tunggu pasien rumah sakit menggunakan FastAPI dan model TensorFlow.

## Menjalankan Lokal

1. Buat virtual environment.
2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan server:

```bash
uvicorn app:app --reload
```

4. Buka dokumentasi API di:

```text
http://127.0.0.1:8000/docs
```

## File Model

Aplikasi memuat model dari `saved_model_smartqueue/` dan scaler dari `scaler.save`.
