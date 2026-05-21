# 🏥 SmartQueue AI — Prediksi Waktu Tunggu Pasien Rumah Sakit

Repositori ini memuat *Capstone Project* untuk memprediksi waktu tunggu pasien rumah sakit. Proyek ini mengimplementasikan algoritma **Deep Learning** menggunakan arsitektur custom dengan framework **TensorFlow/Keras**, yang kemudian di-deploy sebagai REST API menggunakan **FastAPI**.

## 🌟 Fitur Utama (Capstone Criteria Met)

Proyek ini telah memenuhi seluruh kriteria kelulusan utama (Main Quest) dan opsional (Side Quest):

### ✅ Main Quest
1. **Model Deep Learning:** Dibangun menggunakan Keras Functional API.
2. **Custom Layer (`ResidualDenseBlock`):** Mengimplementasikan *skip connections* untuk mengalirkan informasi tanpa modifikasi (mirip arsitektur ResNet) untuk mencegah *vanishing gradient*.
3. **Custom Loss (`WeightedHuberLoss`):** Fungsi loss asimetris yang memberikan penalti 1.1x lebih berat untuk *under-prediction* (karena di rumah sakit, memprediksi waktu tunggu lebih lambat lebih baik daripada terlalu cepat).
4. **Custom Callback (`DetailedTrainingLogger`):** Menyimpan seluruh metrik epoch secara otomatis ke dalam format JSON (`training_log.json`).
5. **Model Export & Inference:** Model disimpan sebagai `best_model.keras` dan diuji langsung pada akhir notebook.

### ✅ Side Quest
1. **REST API (FastAPI):** Aplikasi telah di-deploy secara lokal dengan endpoint `/predict`.
2. **Custom Training Loop (`tf.GradientTape`):** Mendemonstrasikan pelatihan kontrol penuh (manual forward & backward pass) dalam notebook.
3. **TensorBoard Integration:** Callback TensorBoard aktif selama *training*, dan log interaktif disematkan langsung di dalam notebook menggunakan `%tensorboard`.
4. **Kinerja Unggul (Akurasi Tinggi):**
   - R² Score: **> 95%** (Syarat: ≥ 85%)
   - Normalized MAE: **~0.0309** (mewakili margin error murni sekitar 2.6 menit dari rentang target 0-87 menit, bersaing setara dengan algoritma *state-of-the-art* tabular XGBoost).
5. **Visualisasi Komprehensif:** Terdapat visualisasi untuk kurva *training*, perbandingan model, distribusi error, diagram arsitektur model, hingga fitur yang paling berpengaruh (*Feature Importance*).

---

## 📂 Struktur Direktori

```text
smartqueue_project/
├── app/
│   ├── __init__.py
│   └── app.py                        # REST API dengan FastAPI
├── datasets/
│   └── dataset_RS2_final.csv         # Dataset utama (pra-pemrosesan)
├── deployment/
│   └── model/
│       ├── best_model.keras          # Model Deep Learning terbaik yang diekspor
│       ├── feature_scaler.save       # Scaler fitur (StandardScaler)
│       ├── target_scaler.save        # Scaler target (MinMaxScaler)
│       ├── feature_columns.pkl       # Daftar urutan kolom
│       ├── training_log.json         # Log pelatihan (Custom Callback)
│       └── *.png                     # Output visualisasi grafik dari notebook
├── logs/
│   └── fit/                          # TensorBoard event logs
├── notebooks/
│   └── RS2_final_Custom_Model_dan_Custom_Training.ipynb  # NOTEBOOK UTAMA
├── .gitignore
├── requirements.txt
└── README.md
```

> **Catatan:** File-file eksperimen lama berukuran besar dan model legacy telah dipindahkan ke folder `backup/` yang disembunyikan dari Git agar repo ini ringan dan profesional.

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Instalasi Environment
Direkomendasikan menggunakan Virtual Environment.
```bash
python -m venv venv
source venv/bin/activate  # Untuk Mac/Linux
# venv\Scripts\activate   # Untuk Windows

pip install -r requirements.txt
```

### 2. Menjalankan REST API (FastAPI Server)
Pastikan Anda berada di root direktori proyek, lalu jalankan Uvicorn:
```bash
uvicorn app.app:app --reload
```
API akan berjalan di `http://127.0.0.1:8000`.

### 3. Menguji API
Buka dokumentasi interaktif Swagger UI untuk langsung melakukan pengujian:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

**Contoh Request Payload (JSON):**
```json
{
  "umur": 35,
  "jumlah_antrian": 20,
  "jam_kedatangan": 9,
  "asuransi": "BPJS",
  "prioritas": "Sedang",
  "status_pasien": "Rawat Jalan",
  "nama_poli": "Penyakit Dalam"
}
```

*Backend akan secara otomatis memproses fitur-fitur teknis lainnya (seperti `is_peak`, encoding siklus waktu menggunakan sinus/kosinus, dll) berdasarkan tanggal saat request dilakukan.*

---

## 📓 Cara Menjalankan Notebook
Notebook `RS2_final_Custom_Model_dan_Custom_Training.ipynb` dirancang agar dapat dijalankan secara berurutan (*Run All*). 

Pastikan **Jupyter Kernel** Anda diarahkan ke *virtual environment* (`venv`) proyek ini yang telah berisi library `tensorflow`, `xgboost`, `pandas`, dll., sebelum Anda menjalankannya. Semua artefak yang dihasilkan notebook (model `.keras`, *scalers*, gambar plot grafik) akan secara otomatis diperbarui di dalam folder `deployment/model/`.
