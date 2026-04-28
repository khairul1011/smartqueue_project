from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd

# ============================================
# LOAD MODEL DAN SCALER
# ============================================

model = tf.saved_model.load("saved_model_smartqueue")
infer = model.signatures["serving_default"]

scaler = joblib.load("scaler.save")

# ============================================
# FASTAPI INIT
# ============================================

app = FastAPI(
    title="SmartQueue AI API",
    description="REST API prediksi waktu tunggu pasien rumah sakit",
    version="1.0"
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


def estimate_wait_category(wait_time_minutes: float) -> str:
    if wait_time_minutes <= 10:
        return "Low"
    if wait_time_minutes <= 30:
        return "Medium"
    return "High"

# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
def home():
    return {
        "message": "SmartQueue AI API aktif",
        "docs": "/docs"
    }

# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
def predict(data: PatientData):

    # Convert input ke DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Scaling
    scaled_data = scaler.transform(input_data)

    # Convert ke tensor
    tensor_input = tf.convert_to_tensor(
        scaled_data,
        dtype=tf.float32
    )

    # Inference
    prediction = infer(tensor_input)

    # Ambil output tensor pertama
    output = list(prediction.values())[0].numpy()[0][0]

    predicted_wait_time = round(float(output), 2)

    return {
        "predicted_wait_time_minutes": predicted_wait_time,
        "estimated_wait_category": estimate_wait_category(predicted_wait_time),
        "status": "success"
    }
