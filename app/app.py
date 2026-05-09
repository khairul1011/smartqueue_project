from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# ============================================
# LOAD OPTIMIZED MODEL DAN SCALER
# ============================================

# Model hasil optimasi V2
model = tf.keras.models.load_model("experiments/best_model_v2.keras")

# Feature scaler
scaler = joblib.load("experiments/scaler_v2.save")

# Target scaler untuk inverse transform output
target_scaler = joblib.load("models/target_scaler.save")


# ============================================
# FASTAPI INIT
# ============================================

app = FastAPI(
    title="SmartQueue AI API",
    description="REST API prediksi waktu tunggu pasien rumah sakit menggunakan optimized deep learning model",
    version="2.0"
)


# ============================================
# INPUT SCHEMA
# ============================================

class PatientData(BaseModel):
    umur: float
    durasi_layanan: float
    jumlah_antrian: float
    biaya: float
    kepuasan_pasien: float
    jam_kedatangan: float
    is_peak: int
    asuransi_enc: int
    prioritas_enc: int
    status_pasien_enc: int
    nama_poli_enc: int
    day_of_week: int
    month: int
    week_of_year: int
    hour_sin: float
    hour_cos: float


# ============================================
# WAIT CATEGORY ESTIMATOR
# ============================================

def estimate_wait_category(wait_time_minutes: float) -> str:
    if wait_time_minutes <= 10:
        return "Low"
    elif wait_time_minutes <= 30:
        return "Medium"
    else:
        return "High"


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
def home():
    return {
        "message": "SmartQueue AI API Optimized Model aktif",
        "model_version": "v2_optimized",
        "docs": "/docs",
        "status": "running"
    }


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "2.0"
    }


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
def predict(data: PatientData):

    try:
        # Convert input ke DataFrame
        input_data = pd.DataFrame([data.model_dump()])

        # Feature scaling
        scaled_data = scaler.transform(input_data)

        # Prediksi model
        raw_prediction = model.predict(
            scaled_data,
            verbose=0
        )

        # Inverse transform ke menit asli
        predicted_wait_time = target_scaler.inverse_transform(
            raw_prediction.reshape(-1, 1)
        )[0][0]

        # Safety clamp
        predicted_wait_time = max(0, predicted_wait_time)

        # Round output
        predicted_wait_time = round(float(predicted_wait_time), 2)

        return {
            "predicted_wait_time_minutes": predicted_wait_time,
            "estimated_wait_category": estimate_wait_category(predicted_wait_time),
            "model_version": "v2_optimized",
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }