from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class PengaturanPipeline:
    lebar_target: int = 640
    ukuran_blur: tuple[int, int] = (13, 13)
    sigma_blur: float = 3.0
    ukuran_blok: int = 21
    nilai_c: int = 7
    ukuran_kernel_biner: int = 3
    iterasi_biner: int = 1
    luas_minimal: float = 40.0
    luas_maksimal: float = 15000.0
    rasio_aspek_minimal: float = 1.8

    def __post_init__(self) -> None:
        if self.lebar_target <= 0:
            self.lebar_target = 640

        kx, ky = self.ukuran_blur
        kx = max(1, int(kx))
        ky = max(1, int(ky))
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        self.ukuran_blur = (kx, ky)

        blok = max(3, int(self.ukuran_blok))
        if blok % 2 == 0:
            blok += 1
        self.ukuran_blok = blok

        kernel = max(1, int(self.ukuran_kernel_biner))
        if kernel % 2 == 0:
            kernel += 1
        self.ukuran_kernel_biner = kernel

        self.iterasi_biner = max(0, int(self.iterasi_biner))


def resize_lebar_target(citra_bgr: np.ndarray, lebar_target: int) -> np.ndarray:
    tinggi, lebar = citra_bgr.shape[:2]
    if lebar == lebar_target or lebar == 0:
        return citra_bgr.copy()

    rasio = lebar_target / float(lebar)
    tinggi_baru = max(1, int(round(tinggi * rasio)))
    interpolasi = cv2.INTER_AREA if lebar_target < lebar else cv2.INTER_CUBIC
    return cv2.resize(citra_bgr, (lebar_target, tinggi_baru), interpolation=interpolasi)


def proses_citra_telur(
    citra_bgr: np.ndarray,
    pengaturan: PengaturanPipeline | None = None,
) -> dict[str, Any]:
    if citra_bgr is None or citra_bgr.size == 0:
        raise ValueError("Citra input kosong atau tidak valid.")

    cfg = pengaturan or PengaturanPipeline()

    # 1. Resize proporsional
    citra_resized = resize_lebar_target(citra_bgr, cfg.lebar_target)

    # 2. Konversi Grayscale
    grayscale = cv2.cvtColor(citra_resized, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian Blur
    blur = cv2.GaussianBlur(grayscale, cfg.ukuran_blur, cfg.sigma_blur)

    # 4. Adaptive Thresholding (THRESH_BINARY_INV)
    adaptive_thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.ukuran_blok,
        cfg.nilai_c,
    )

    # 5. Penyambungan celah kecil pada citra biner sebelum analisis kontur
    if cfg.iterasi_biner > 0 and cfg.ukuran_kernel_biner > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.ukuran_kernel_biner, cfg.ukuran_kernel_biner),
        )
        citra_biner_bersih = cv2.morphologyEx(
            adaptive_thresh,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=cfg.iterasi_biner,
        )
    else:
        citra_biner_bersih = adaptive_thresh.copy()

    # 6. Ekstraksi dan penyaringan geometri kontur
    kontur_list, _ = cv2.findContours(
        citra_biner_bersih,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    hasil_visual = citra_resized.copy()
    kandidat_lolos: list[dict[str, Any]] = []

    for idx, cnt in enumerate(kontur_list, start=1):
        luas = float(cv2.contourArea(cnt))
        if luas < cfg.luas_minimal or luas > cfg.luas_maksimal:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        sisi_panjang = max(w, h)
        sisi_pendek = max(min(w, h), 1)
        rasio_aspek = round(sisi_panjang / float(sisi_pendek), 2)

        if rasio_aspek < cfg.rasio_aspek_minimal:
            continue

        keliling = round(float(cv2.arcLength(cnt, True)), 2)
        kandidat_lolos.append(
            {
                "id": len(kandidat_lolos) + 1,
                "kontur_ke": idx,
                "luas_px2": round(luas, 2),
                "keliling_px": keliling,
                "rasio_aspek": rasio_aspek,
                "x": int(x),
                "y": int(y),
                "lebar_px": int(w),
                "tinggi_px": int(h),
            }
        )

        cv2.rectangle(hasil_visual, (x, y), (x + w, y + h), (0, 0, 255), 2)

    jumlah_kandidat = len(kandidat_lolos)
    prediksi = "RETAK" if jumlah_kandidat > 0 else "MULUS"

    return {
        "prediksi": prediksi,
        "jumlah_kandidat": jumlah_kandidat,
        "kandidat": kandidat_lolos,
        "tahapan": {
            "asli_resized": citra_resized,
            "grayscale": grayscale,
            "gaussian_blur": blur,
            "adaptive_threshold": adaptive_thresh,
            "citra_biner_bersih": citra_biner_bersih,
            "hasil_deteksi": hasil_visual,
        },
    }