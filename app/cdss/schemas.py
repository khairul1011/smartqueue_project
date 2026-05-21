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


class PenyakitRekomendasi(BaseModel):
    """Detail satu rekomendasi penyakit."""

    nama_penyakit: str = Field(description="Nama penyakit yang direkomendasikan.")
    tingkat_kesesuaian: str = Field(
        description="Tingkat kesesuaian: 'Tinggi', 'Sedang', atau 'Rendah'."
    )
    penjelasan: str = Field(
        description="Penjelasan singkat mengapa penyakit ini direkomendasikan."
    )
    pemeriksaan_lanjutan: List[str] = Field(
        description="Daftar saran pemeriksaan lanjutan yang relevan."
    )


class CDSSResponse(BaseModel):
    """Schema response rekomendasi penyakit."""

    rekomendasi: List[PenyakitRekomendasi] = Field(
        description="Daftar rekomendasi penyakit (maksimal 5)."
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
