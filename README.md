# SmartQueue AI — Prediksi Waktu Tunggu Pasien Rumah Sakit

Repositori ini memuat *Capstone Project* untuk memprediksi waktu tunggu pasien rumah sakit. Proyek ini mengimplementasikan algoritma Deep Learning menggunakan arsitektur custom dengan framework TensorFlow/Keras, yang kemudian di-deploy sebagai REST API menggunakan FastAPI.

## Fitur Utama (Capstone Criteria)

Proyek ini telah memenuhi seluruh kriteria kelulusan utama (Main Quest) dan opsional (Side Quest):

### Kriteria Utama (Main Quest)
1. **Model Deep Learning:** Dibangun menggunakan Keras Functional API.
2. **Custom Layer (`ResidualDenseBlock`):** Mengimplementasikan *skip connections* untuk mengalirkan informasi tanpa modifikasi (mengadaptasi arsitektur ResNet) guna mencegah *vanishing gradient*.
3. **Custom Loss (`WeightedHuberLoss`):** Fungsi loss asimetris yang memberikan penalti 1.1x lebih berat untuk *under-prediction* (dalam konteks rumah sakit, memprediksi waktu tunggu lebih lambat dinilai lebih aman daripada terlalu cepat).
4. **Custom Callback (`DetailedTrainingLogger`):** Menyimpan seluruh metrik per epoch secara otomatis ke dalam format JSON (`training_log.json`).
5. **Model Export & Inference:** Model disimpan sebagai `best_model.keras` dan diuji secara langsung pada bagian akhir notebook.

### Kriteria Opsional (Side Quest)
1. **REST API (FastAPI):** Aplikasi telah di-deploy secara lokal dengan menyediakan endpoint `/predict`.
2. **Custom Training Loop (`tf.GradientTape`):** Mendemonstrasikan pelatihan dengan kontrol penuh (pengaturan manual *forward* dan *backward pass*) di dalam notebook.
3. **TensorBoard Integration:** Callback TensorBoard aktif selama masa *training*, dan log interaktif disematkan langsung di dalam notebook menggunakan `%tensorboard`.
4. **Kinerja Unggul:**
   - R² Score: **> 95%** (Syarat kelulusan: ≥ 85%)
   - Normalized MAE: **~0.0309** (mewakili margin error murni sekitar 2.6 menit dari rentang target 0-87 menit, memiliki performa yang setara dengan algoritma *state-of-the-art* tabular seperti XGBoost).
5. **Visualisasi Komprehensif:** Terdapat visualisasi detail untuk kurva *training*, perbandingan antar model, distribusi error, diagram arsitektur model, hingga analisis fitur yang paling berpengaruh (*Feature Importance*).

---

## Struktur Direktori

```text
smartqueue_project/
├── app/
│   ├── __init__.py
│   ├── app.py                        # REST API utama (prediksi waktu tunggu + routing)
│   ├── config.py                     # Konfigurasi terpusat (API keys, env vars)
│   └── cdss/                         # Modul CDSS (Clinical Decision Support System)
│       ├── __init__.py
│       ├── router.py                 # Endpoint /cdss/recommend & /cdss/health
│       ├── schemas.py                # Pydantic models request/response CDSS
│       ├── prompt.py                 # Prompt template untuk Gemini API
│       └── service.py                # Logic integrasi Gemini API
├── datasets/
│   └── dataset_RS2_final.csv         # Dataset utama (telah diproses)
├── deployment/
│   └── model/
│       ├── best_model.keras          # Model Deep Learning terbaik yang diekspor
│       ├── feature_scaler.save       # Scaler fitur (StandardScaler)
│       ├── target_scaler.save        # Scaler target (MinMaxScaler)
│       ├── feature_columns.pkl       # Daftar dan urutan kolom fitur
│       ├── training_log.json         # Log pelatihan model (Custom Callback)
│       └── *.png                     # Output visualisasi grafik dari notebook
├── logs/
│   └── fit/                          # TensorBoard event logs
├── notebooks/
│   └── RS2_final_Custom_Model_dan_Custom_Training.ipynb  # NOTEBOOK UTAMA
├── .env                              # Environment variables (API keys, TIDAK di-commit)
├── .gitignore
├── requirements.txt
└── README.md
```

> **Catatan:** File-file eksperimen lama berukuran besar dan model *legacy* telah dipindahkan ke folder backup yang diabaikan (*ignored*) oleh Git agar ukuran repositori tetap ringan dan profesional.

---

## Cara Menjalankan Aplikasi

### 1. Instalasi Environment
Sangat direkomendasikan untuk menggunakan *Virtual Environment*.
```bash
python -m venv venv
source venv/bin/activate  # Untuk pengguna Mac/Linux
# venv\Scripts\activate   # Untuk pengguna Windows

pip install -r requirements.txt
```

### 2. Menjalankan REST API (FastAPI Server)
Pastikan terminal berada di *root* direktori proyek, kemudian jalankan Uvicorn:
```bash
uvicorn app.app:app --reload
```
API akan berjalan di `http://127.0.0.1:8000`.

### 3. Menguji API
Buka dokumentasi interaktif Swagger UI pada browser untuk melakukan pengujian langsung:
**http://127.0.0.1:8000/docs**

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

*Sistem backend akan secara otomatis memproses kebutuhan *feature engineering* teknis lainnya (seperti penentuan status `is_peak`, encoding siklus waktu periodik menggunakan nilai sinus/kosinus, dll.) berdasarkan tanggal sistem saat permintaan diterima.*

---

## Fitur Tambahan: CDSS — Clinical Decision Support System

Sistem ini dilengkapi fitur **CDSS** untuk membantu dokter mendapatkan rekomendasi kemungkinan penyakit (maksimal 3) beserta saran pemeriksaan lanjutan berdasarkan input gejala pasien. Fitur ini didukung oleh **Google Gemini API** yang telah dioptimasi untuk efisiensi token.

### Keunggulan Sistem CDSS Kami
- **Token Efficient:** Membatasi output menjadi maksimal 3 penyakit teratas dan penjelasan singkat (1-2 kalimat) untuk menghemat penggunaan token/biaya secara signifikan.
- **Auto Model Fallback:** Sistem tahan banting terhadap error limit/kuota (*Rate Limit 429*). Jika model utama (misal `gemini-2.5-flash`) gagal karena kuota harian, sistem akan **otomatis** dan diam-diam beralih ke model cadangan (`gemini-flash-latest`) sehingga pengguna tidak pernah mengalami *downtime*.

### Setup Lingkungan (Environment)

1. Kunjungi [Google AI Studio](https://aistudio.google.com/apikey) dan buat API key gratis.
2. Konfigurasikan `.env` di root project seperti berikut:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

### Endpoint CDSS

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/cdss/recommend` | Rekomendasi penyakit dan penunjang diagnostik |
| `GET` | `/cdss/health` | Status konektivitas dengan server Google Gemini |

**Contoh Request Payload CDSS (JSON):**
```json
{
  "gejala": "demam tinggi sudah 3 hari, batuk kering, sesak napas, nyeri dada saat bernapas",
  "umur": 45,
  "jenis_kelamin": "L"
}
```

**Contoh Response Payload CDSS (JSON):**
```json
{
  "gejala_teridentifikasi": [
    "demam tinggi", "batuk kering", "sesak napas", "nyeri dada"
  ],
  "kandidat_diagnosis": [
    {
      "nama_penyakit": "Pneumonia",
      "tingkat_urgensi": "HIGH",
      "confidence": 85,
      "departemen": "PARU",
      "penjelasan": "Gejala demam tinggi disertai batuk kering dan sesak napas sangat mengarah pada infeksi paru.",
      "pemeriksaan_lanjutan": ["Rontgen Thorax", "Cek Darah Lengkap"]
    }
  ],
  "catatan_medis": "Segera lakukan observasi saturasi oksigen pasien.",
  "disclaimer": "Hasil ini merupakan rekomendasi berbasis AI...",
  "status": "success"
}
```

> **⚠️ Disclaimer:** Fitur CDSS merupakan alat bantu berbasis AI dan **BUKAN** pengganti diagnosis medis. Keputusan klinis tetap sepenuhnya berada di tangan dokter yang merawat.

---

## Cara Menjalankan Notebook
Notebook `RS2_final_Custom_Model_dan_Custom_Training.ipynb` dirancang agar dapat dieksekusi secara berurutan dari awal hingga akhir (*Run All*). 

Pastikan pengaturan **Jupyter Kernel** Anda telah diarahkan ke *virtual environment* (`venv`) proyek ini yang memuat semua pustaka prasyarat (seperti `tensorflow`, `xgboost`, `pandas`). Semua artefak yang dihasilkan dari eksekusi notebook (termasuk model `.keras`, *scalers*, dan output gambar grafik) akan diperbarui secara otomatis dan disalin ke dalam direktori `deployment/model/`.

