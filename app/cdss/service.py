"""Service layer — integrasi dengan Google Gemini API untuk CDSS."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import google.generativeai as genai

from app.cdss.prompt import SYSTEM_INSTRUCTION, build_user_prompt
from app.cdss.schemas import CDSSResponse, KandidatDiagnosis
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi Gemini
# ---------------------------------------------------------------------------

_models: Dict[str, genai.GenerativeModel] = {}


def _get_model(model_name: str) -> genai.GenerativeModel:
    """Lazy-init Gemini model berdasarkan nama model."""
    if model_name not in _models:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _models[model_name] = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                response_mime_type="application/json",
            ),
        )
    return _models[model_name]


# ---------------------------------------------------------------------------
# Parsing & Validasi Response
# ---------------------------------------------------------------------------


def _parse_gemini_response(raw_text: str) -> Dict[str, Any]:
    """Parse JSON response dari Gemini, handle edge cases."""
    text = raw_text.strip()

    # Hapus markdown code fences jika ada
    if text.startswith("```"):
        lines = text.split("\n")
        # Buang baris pertama (```json) dan terakhir (```)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    return json.loads(text)


def _build_response(data: Dict[str, Any]) -> CDSSResponse:
    """Konversi dict hasil parsing menjadi CDSSResponse tervalidasi."""
    kandidat_list = []
    for item in data.get("kandidat_diagnosis", []):
        kandidat_list.append(
            KandidatDiagnosis(
                nama_penyakit=item.get("nama_penyakit", "Tidak diketahui"),
                tingkat_urgensi=item.get("tingkat_urgensi", "LOW"),
                confidence=int(item.get("confidence", 0)),
                departemen=item.get("departemen", "UMUM"),
                penjelasan=item.get("penjelasan", ""),
                pemeriksaan_lanjutan=item.get("pemeriksaan_lanjutan", []),
            )
        )

    return CDSSResponse(
        gejala_teridentifikasi=data.get("gejala_teridentifikasi", []),
        kandidat_diagnosis=kandidat_list[:3],  # Maksimal 3
        catatan_medis=data.get("catatan_medis", "Tidak ada catatan tambahan."),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_disease_recommendation(
    gejala: str,
    umur: Optional[int] = None,
    jenis_kelamin: Optional[str] = None,
) -> CDSSResponse:
    """
    Kirim gejala ke Gemini API dan kembalikan rekomendasi penyakit terstruktur.

    Raises:
        ConnectionError: Jika Gemini API tidak bisa dihubungi.
        ValueError: Jika response dari Gemini tidak bisa di-parse.
        RuntimeError: Jika API key belum dikonfigurasi.
    """
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY belum dikonfigurasi. "
            "Silakan set API key di file .env"
        )

    user_prompt = build_user_prompt(gejala, umur, jenis_kelamin)
    
    primary_model = settings.GEMINI_MODEL
    fallback_model = "gemini-flash-latest"
    
    # Daftar model yang akan dicoba berurutan
    models_to_try = [primary_model]
    if primary_model != fallback_model:
        models_to_try.append(fallback_model)

    response = None
    last_error = None

    for m_name in models_to_try:
        try:
            model = _get_model(m_name)
            logger.info("Mencoba menghubungi Gemini API dengan model: %s", m_name)
            response = await model.generate_content_async(user_prompt)
            break  # Jika sukses, keluar dari loop
        except Exception as exc:
            logger.warning("Model %s gagal: %s. Mencoba model berikutnya (jika ada)...", m_name, exc)
            last_error = exc
            continue

    if response is None:
        logger.error("Semua model (utama & fallback) gagal. Error terakhir: %s", last_error)
        raise ConnectionError(
            f"Gagal menghubungi Gemini API (semua model gagal): {last_error}"
        ) from last_error

    if not response.text:
        raise ValueError("Gemini API mengembalikan response kosong.")

    try:
        data = _parse_gemini_response(response.text)
    except json.JSONDecodeError as exc:
        logger.error(
            "Gagal parse JSON dari Gemini: %s\nRaw: %s",
            exc,
            response.text[:500],
        )
        raise ValueError(
            "Response dari Gemini tidak dalam format JSON yang valid."
        ) from exc

    return _build_response(data)


def check_api_configured() -> bool:
    """Cek apakah Gemini API key sudah dikonfigurasi."""
    return bool(
        settings.GEMINI_API_KEY
        and settings.GEMINI_API_KEY != "your_api_key_here"
    )
