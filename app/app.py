import json
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "deployment" / "model"


# Keeps models saved by newer Keras versions loadable in older runtime images.
def remove_empty_quantization_config(config):
    if isinstance(config, dict):
        if config.get("quantization_config") is None:
            config.pop("quantization_config", None)

        for value in config.values():
            remove_empty_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            remove_empty_quantization_config(item)


def read_keras_config(model_path: Path):
    with zipfile.ZipFile(model_path, "r") as model_archive:
        return json.loads(model_archive.read("config.json"))


def add_layer_from_config(model, layer_config):
    class_name = layer_config["class_name"]
    config = layer_config["config"]

    if class_name == "InputLayer":
        batch_shape = config.get("batch_shape") or config.get("batch_input_shape")
        input_shape = tuple(batch_shape[1:]) if batch_shape else None
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name=config.get("name")))
    elif class_name == "Dense":
        model.add(
            tf.keras.layers.Dense(
                units=config["units"],
                activation=config.get("activation"),
                use_bias=config.get("use_bias", True),
                name=config.get("name")
            )
        )
    elif class_name == "BatchNormalization":
        model.add(
            tf.keras.layers.BatchNormalization(
                axis=config.get("axis", -1),
                momentum=config.get("momentum", 0.99),
                epsilon=config.get("epsilon", 0.001),
                center=config.get("center", True),
                scale=config.get("scale", True),
                name=config.get("name")
            )
        )
    elif class_name == "Dropout":
        model.add(
            tf.keras.layers.Dropout(
                rate=config["rate"],
                name=config.get("name")
            )
        )
    else:
        raise ValueError(f"Unsupported model layer: {class_name}")


def load_sequential_model_from_weights(model_path: Path):
    model_config = read_keras_config(model_path)
    model = tf.keras.Sequential(name=model_config["config"].get("name"))

    for layer_config in model_config["config"]["layers"]:
        add_layer_from_config(model, layer_config)

    with tempfile.TemporaryDirectory() as temp_dir:
        weights_path = Path(temp_dir) / "model.weights.h5"

        with zipfile.ZipFile(model_path, "r") as model_archive:
            weights_path.write_bytes(model_archive.read("model.weights.h5"))

        model.load_weights(weights_path)

    return model


def load_prediction_model(model_path: Path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as error:
        if "quantization_config" not in str(error):
            return load_sequential_model_from_weights(model_path)

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

            try:
                return tf.keras.models.load_model(compatible_model_path, compile=False)
            except (TypeError, ValueError):
                return load_sequential_model_from_weights(model_path)


model = load_prediction_model(MODEL_DIR / "best_model.keras")
scaler = joblib.load(MODEL_DIR / "feature_scaler.save")
target_scaler = joblib.load(MODEL_DIR / "target_scaler.save")

app = FastAPI(
    title="SmartQueue AI API",
    description="REST API prediksi waktu tunggu pasien rumah sakit menggunakan optimized deep learning model",
    version="2.0"
)


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
    elif wait_time_minutes <= 30:
        return "Medium"
    else:
        return "High"


@app.get("/")
def home():
    return {
        "message": "SmartQueue AI API Optimized Model aktif",
        "model_version": "v2_optimized",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "2.0"
    }


@app.post("/predict")
def predict(data: PatientData):
    try:
        input_data = pd.DataFrame([data.model_dump()])
        scaled_data = scaler.transform(input_data)
        raw_prediction = model.predict(scaled_data, verbose=0)
        predicted_wait_time = target_scaler.inverse_transform(
            raw_prediction.reshape(-1, 1)
        )[0][0]
        predicted_wait_time = max(0, predicted_wait_time)
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
