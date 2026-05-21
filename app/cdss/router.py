"""FastAPI router untuk endpoint CDSS."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.cdss.schemas import CDSSRequest, CDSSResponse
from app.cdss.service import check_api_configured, get_disease_recommendation

logger = logging.getLogger(__name__)

cdss_router = APIRouter(
    prefix="/cdss",
    tags=["CDSS - Clinical Decision Support System"],
)


@cdss_router.post(
    "/recommend",
    response_model=CDSSResponse,
    summary="Rekomendasi penyakit berdasarkan gejala",
    description=(
        "Menerima deskripsi gejala pasien dalam teks bebas dan mengembalikan "
        "daftar rekomendasi kemungkinan penyakit beserta saran pemeriksaan lanjutan. "
        "Fitur ini menggunakan AI sebagai alat bantu dan BUKAN pengganti diagnosis dokter."
    ),
)
async def recommend_disease(payload: CDSSRequest) -> CDSSResponse:
    """Endpoint utama CDSS — rekomendasi penyakit berdasarkan gejala."""
    try:
        result = await get_disease_recommendation(
            gejala=payload.gejala,
            umur=payload.umur,
            jenis_kelamin=payload.jenis_kelamin,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Error tidak terduga pada CDSS: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan internal: {exc}",
        ) from exc

    return result


@cdss_router.get(
    "/health",
    summary="Health check CDSS",
    description="Mengecek status konfigurasi Gemini API untuk fitur CDSS.",
)
def cdss_health_check():
    """Health check untuk memastikan CDSS siap digunakan."""
    api_configured = check_api_configured()
    return {
        "status": "healthy" if api_configured else "not_configured",
        "gemini_api_configured": api_configured,
        "message": (
            "CDSS siap digunakan."
            if api_configured
            else "GEMINI_API_KEY belum dikonfigurasi di file .env"
        ),
    }
