"""Pydantic schemas untuk request dan response CDSS."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CDSSRequest(BaseModel):
    """Schema request untuk rekomendasi penyakit berdasarkan gejala."""

    gejala: str = Field(
        min_length=3,
        description="Deskripsi gejala pasien dalam teks bebas (Bahasa Indonesia).",
    )
    umur: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Usia pasien (opsional, untuk konteks diagnosis).",
    )
    jenis_kelamin: Optional[str] = Field(
        default=None,
        description="Jenis kelamin pasien: 'L' atau 'P' (opsional).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gejala": "demam tinggi sudah 3 hari, batuk kering, sesak napas, nyeri dada saat bernapas",
                "umur": 45,
                "jenis_kelamin": "L",
            }
        }
    )


class KandidatDiagnosis(BaseModel):
    """Detail satu kandidat diagnosis."""

    nama_penyakit: str = Field(description="Nama penyakit yang direkomendasikan.")
    tingkat_urgensi: str = Field(
        description="Tingkat urgensi: 'LOW', 'MEDIUM', atau 'HIGH'."
    )
    confidence: int = Field(
        description="Tingkat keyakinan prediksi dalam bentuk persentase (1-100)."
    )
    departemen: str = Field(
        description="Departemen terkait, contoh: 'UMUM', 'BEDAH', 'PENYAKIT DALAM'."
    )
    penjelasan: str = Field(
        description="Penjelasan singkat mengapa penyakit ini direkomendasikan."
    )
    pemeriksaan_lanjutan: List[str] = Field(
        default=[],
        description="Daftar saran pemeriksaan lanjutan yang relevan."
    )


class CDSSResponse(BaseModel):
    """Schema response rekomendasi penyakit."""

    gejala_teridentifikasi: List[str] = Field(
        description="Daftar frasa gejala utama yang teridentifikasi dari input."
    )
    kandidat_diagnosis: List[KandidatDiagnosis] = Field(
        description="Daftar kandidat diagnosis (maksimal 3)."
    )
    catatan_medis: str = Field(
        description="Catatan dan saran medis tambahan dari sistem."
    )
    disclaimer: str = Field(
        default=(
            "Hasil ini merupakan rekomendasi berbasis AI dan BUKAN diagnosis medis. "
            "Keputusan klinis tetap sepenuhnya berada di tangan dokter yang menangani."
        ),
        description="Disclaimer wajib bahwa ini bukan diagnosis.",
    )
    status: str = "success"
