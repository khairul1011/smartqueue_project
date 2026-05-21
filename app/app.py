import json
import math
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Union
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "deployment" / "model"
TIMEZONE = ZoneInfo("Asia/Jakarta")
MODEL_VERSION = "latest_residual_model"

ASURANSI_MAPPING = {
    "bpjs": 0,
    "umum": 1,
}

PRIORITAS_MAPPING = {
    "normal": 0,
    "rendah": 0,
    "sedang": 0,
    "urgent": 1,
    "tinggi": 1,
    "darurat": 1,
}

STATUS_PASIEN_MAPPING = {
    "rawat inap": 0,
    "inap": 0,
    "rawat jalan": 1,
    "jalan": 1,
}

NAMA_POLI_MAPPING = {
    "anak": 0,
    "poli anak": 0,
    "gigi": 1,
    "poli gigi": 1,
    "jantung": 2,
    "poli jantung": 2,
    "mata": 3,
    "poli mata": 3,
    "penyakit dalam": 4,
    "poli penyakit dalam": 4,
    "umum": 5,
    "poli umum": 5,
}


class PredictionRequest(BaseModel):
    umur: float = Field(gt=0, le=120)
    jumlah_antrian: float = Field(ge=0)
    jam_kedatangan: int = Field(ge=0, le=23)
    asuransi: str = Field(min_length=1)
    prioritas: str = Field(min_length=1)
    status_pasien: str = Field(min_length=1)
    nama_poli: str = Field(min_length=1)
    tanggal: Optional[date] = Field(
        default=None,
        description="Opsional. Jika tidak dikirim, backend memakai tanggal hari ini.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "umur": 35,
                "jumlah_antrian": 20,
                "jam_kedatangan": 9,
                "asuransi": "BPJS",
                "prioritas": "Sedang",
                "status_pasien": "Rawat Jalan",
                "nama_poli": "Penyakit Dalam",
            }
        }
    )

    @field_validator("asuransi", "prioritas", "status_pasien", "nama_poli")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field tidak boleh kosong")
        return value


class PredictionResponse(BaseModel):
    predicted_waiting_time_minutes: float
    kategori_waktu_tunggu: str
    status: str = "success"


@tf.keras.utils.register_keras_serializable(package="Custom")
class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = tf.keras.layers.Dense(units, activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(units, activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.dense1.build(input_shape)
        hidden_shape = self.dense1.compute_output_shape(input_shape)
        self.bn1.build(hidden_shape)
        self.dense2.build(hidden_shape)
        self.bn2.build(hidden_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        residual = x
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class WeightedHuberLoss(tf.keras.losses.Loss):
    def __init__(self, delta=1.0, under_weight=1.5, name="weighted_huber_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.under_weight = under_weight

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic**2 + self.delta * linear
        weight = tf.where(error > 0, self.under_weight, 1.0)
        return tf.reduce_mean(weight * base_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"delta": self.delta, "under_weight": self.under_weight})
        return config


def remove_empty_quantization_config(config):
    if isinstance(config, dict):
        if config.get("quantization_config") is None:
            config.pop("quantization_config", None)

        for value in config.values():
            remove_empty_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            remove_empty_quantization_config(item)


def load_prediction_model(model_path: Path):
    custom_objects = {
        "ResidualDenseBlock": ResidualDenseBlock,
        "WeightedHuberLoss": WeightedHuberLoss,
    }

    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )
    except TypeError as error:
        if "quantization_config" not in str(error):
            raise

        with tempfile.TemporaryDirectory() as temp_dir:
            compatible_model_path = Path(temp_dir) / model_path.name

            with zipfile.ZipFile(model_path, "r") as source:
                with zipfile.ZipFile(compatible_model_path, "w") as target:
                    for file_info in source.infolist():
                        data = source.read(file_info.filename)

                        if file_info.filename == "config.json":
                            config = json.loads(data)
                            remove_empty_quantization_config(config)
                            data = json.dumps(config).encode("utf-8")

                        target.writestr(file_info, data)

            return tf.keras.models.load_model(
                compatible_model_path,
                compile=False,
                custom_objects=custom_objects,
            )


model = load_prediction_model(MODEL_DIR / "best_model.keras")
scaler = joblib.load(MODEL_DIR / "feature_scaler.save")
target_scaler = joblib.load(MODEL_DIR / "target_scaler.save")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

app = FastAPI(
    title="SmartQueue AI API",
    description=(
        "REST API prediksi waktu tunggu pasien rumah sakit "
        "dan Clinical Decision Support System (CDSS)."
    ),
    version="4.1",
)

# --- CDSS Router ---
from app.cdss.router import cdss_router  # noqa: E402

app.include_router(cdss_router)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def encode_category(field_name: str, value: str, mapping: Dict[str, int]) -> int:
    normalized_value = normalize_text(value)
    if normalized_value not in mapping:
        allowed_values = ", ".join(sorted(mapping))
        raise ValueError(
            f"Nilai '{value}' tidak valid untuk {field_name}. "
            f"Nilai yang didukung: {allowed_values}."
        )

    return mapping[normalized_value]


def resolve_inference_date(requested_date: Optional[date]) -> date:
    return requested_date or datetime.now(TIMEZONE).date()


def generate_time_features(
    jam_kedatangan: int,
    inference_date: date,
) -> Dict[str, Union[float, int]]:
    return {
        "is_peak": int(8 <= jam_kedatangan <= 11),
        "day_of_week": inference_date.weekday(),
        "month": inference_date.month,
        "week_of_year": int(inference_date.isocalendar().week),
        "hour_sin": math.sin(2 * math.pi * jam_kedatangan / 24),
        "hour_cos": math.cos(2 * math.pi * jam_kedatangan / 24),
    }


def preprocess_request(payload: PredictionRequest) -> pd.DataFrame:
    inference_date = resolve_inference_date(payload.tanggal)

    features = {
        "umur": payload.umur,
        "jumlah_antrian": payload.jumlah_antrian,
        "jam_kedatangan": payload.jam_kedatangan,
        "asuransi_enc": encode_category(
            "asuransi",
            payload.asuransi,
            ASURANSI_MAPPING,
        ),
        "prioritas_enc": encode_category(
            "prioritas",
            payload.prioritas,
            PRIORITAS_MAPPING,
        ),
        "status_pasien_enc": encode_category(
            "status_pasien",
            payload.status_pasien,
            STATUS_PASIEN_MAPPING,
        ),
        "nama_poli_enc": encode_category(
            "nama_poli",
            payload.nama_poli,
            NAMA_POLI_MAPPING,
        ),
    }
    features.update(generate_time_features(payload.jam_kedatangan, inference_date))

    input_data = pd.DataFrame([features]).reindex(columns=feature_columns)
    missing_features = input_data.columns[input_data.isna().any()].tolist()
    if missing_features:
        raise RuntimeError(f"Feature model belum lengkap: {missing_features}")

    return input_data


def get_kategori_waktu_tunggu(minutes: float) -> str:
    if minutes <= 10:
        return "Cepat"
    elif minutes <= 25:
        return "Normal"
    elif minutes <= 45:
        return "Lama"
    else:
        return "Sangat Lama"


def predict_waiting_time(input_data: pd.DataFrame) -> float:
    scaled_data = scaler.transform(input_data)
    scaled_prediction = model.predict(scaled_data, verbose=0)
    predicted_minutes = target_scaler.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    )[0][0]

    return round(max(0.0, float(predicted_minutes)), 2)


@app.get("/")
def home():
    return {
        "message": "SmartQueue AI API aktif",
        "model_version": MODEL_VERSION,
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "feature_count": len(feature_columns),
        "version": "4.0",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        input_data = preprocess_request(payload)
        predicted_minutes = predict_waiting_time(input_data)
        kategori = get_kategori_waktu_tunggu(predicted_minutes)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal melakukan prediksi: {error}",
        ) from error

    return PredictionResponse(
        predicted_waiting_time_minutes=predicted_minutes,
        kategori_waktu_tunggu=kategori
    )
