import cv2
import numpy as np
import streamlit as st

from pipeline_telur import PengaturanPipeline, proses_citra_telur


def bgr_ke_rgb(citra_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="Identifikasi Keretakan Telur",
        page_icon="🥚",
        layout="wide",
    )

    st.title("🥚 Identifikasi Keretakan Telur")
    st.caption(
        "Berbasis Pengolahan Citra Digital (Resize, Grayscale, Gaussian Blur, "
        "Thresholding & Analisis Kontur)"
    )

    with st.sidebar:
        st.header("⚙️ Pengaturan Parameter")
        st.caption("Segmentasi: **Adaptive Thresholding** (`THRESH_BINARY_INV`)")

        ukuran_blok = st.slider(
            "Ukuran Blok Adaptive (Ganjil)",
            min_value=3,
            max_value=51,
            value=21,
            step=2,
        )
        nilai_c = st.slider(
            "Nilai C Adaptive",
            min_value=2,
            max_value=25,
            value=7,
            step=1,
        )

        st.divider()
        st.subheader("📐 Filter Geometri Kontur")
        luas_minimal = st.number_input(
            "Luas Minimal (px²)",
            min_value=0.0,
            max_value=50000.0,
            value=40.0,
            step=5.0,
        )
        rasio_aspek_minimal = st.slider(
            "Rasio Aspek Minimal",
            min_value=1.0,
            max_value=10.0,
            value=1.8,
            step=0.1,
        )

    berkas_unggah = st.file_uploader(
        "Unggah foto telur (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if berkas_unggah is None:
        st.info("Silakan unggah satu foto telur terlebih dahulu untuk melihat hasil deteksi.")
        return

    file_bytes = np.asarray(bytearray(berkas_unggah.read()), dtype=np.uint8)
    citra_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if citra_bgr is None:
        st.error("Berkas citra tidak dapat dibaca. Coba unggah file JPG, JPEG, atau PNG lain.")
        return

    pengaturan = PengaturanPipeline(
        ukuran_blok=ukuran_blok,
        nilai_c=nilai_c,
        luas_minimal=luas_minimal,
        rasio_aspek_minimal=rasio_aspek_minimal,
    )

    hasil = proses_citra_telur(citra_bgr, pengaturan)
    tahapan = hasil["tahapan"]

    st.subheader("Hasil Deteksi")
    kol_status, kol_gambar, kol_info = st.columns([1.1, 1.4, 1.5])

    with kol_status:
        if hasil["prediksi"] == "RETAK":
            st.error(f"Prediksi: **RETAK** ({hasil['jumlah_kandidat']} kandidat lolos)")
        else:
            st.success("Prediksi: **MULUS** (0 kandidat lolos)")

    with kol_gambar:
        st.image(
            bgr_ke_rgb(tahapan["hasil_deteksi"]),
            caption="Citra Hasil Deteksi (Kotak Merah = Kandidat Retakan)",
            width=360,
        )

    with kol_info:
        st.markdown("**Ringkasan Parameter Aktif**")
        st.write(f"- Ukuran Blok: `{ukuran_blok}`")
        st.write(f"- Nilai C: `{nilai_c}`")
        st.write(f"- Luas Minimal: `{luas_minimal} px²`")
        st.write(f"- Rasio Aspek Minimal: `{rasio_aspek_minimal}`")

    st.subheader("Rincian Kandidat Kontur Lolos Filter")
    if hasil["kandidat"]:
        st.dataframe(hasil["kandidat"], use_container_width=True)
    else:
        st.write("Tidak ada kontur yang memenuhi batas luas minimal dan rasio aspek minimal.")

    st.subheader("Visualisasi Tahapan Pengolahan Citra")
    baris1_k1, baris1_k2, baris1_k3 = st.columns(3)
    with baris1_k1:
        st.image(
            bgr_ke_rgb(tahapan["asli_resized"]),
            caption="1. Citra Resize (Lebar 640 px)",
            use_container_width=True,
        )
    with baris1_k2:
        st.image(
            tahapan["grayscale"],
            caption="2. Grayscale",
            clamp=True,
            use_container_width=True,
        )
    with baris1_k3:
        st.image(
            tahapan["gaussian_blur"],
            caption="3. Gaussian Blur",
            clamp=True,
            use_container_width=True,
        )

    baris2_k1, baris2_k2 = st.columns(2)
    with baris2_k1:
        st.image(
            tahapan["adaptive_threshold"],
            caption="4. Adaptive Threshold (THRESH_BINARY_INV)",
            clamp=True,
            use_container_width=True,
        )
    with baris2_k2:
        st.image(
            tahapan["citra_biner_bersih"],
            caption="5. Citra Biner Sebelum Analisis Kontur",
            clamp=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()