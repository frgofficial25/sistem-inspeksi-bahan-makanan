
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
import csv
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Sistem Inspeksi Kualitas Bahan Makanan", page_icon="🥕", layout="wide")

CLASS_NAMES = [
    'fresh_carrot', 'fresh_chilli', 'fresh_eggplant', 'fresh_onion',
    'fresh_pepper', 'fresh_potato', 'fresh_pumpkin', 'fresh_zucchini',
    'rotten_carrot', 'rotten_chilli', 'rotten_eggplant', 'rotten_onion',
    'rotten_pepper', 'rotten_potato', 'rotten_pumpkin', 'rotten_zucchini'
]

MODEL_FILE_ID = "1EZlSyHbPRFzmPKcPojf9dlGYNw-Txubk"
MODEL_PATH = "model_klasifikasi_best.keras"
LOG_PATH = "log_aktivitas.csv"


def catat_log(level, pesan):
    file_baru = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if file_baru:
            writer.writerow(['waktu', 'level', 'pesan'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level, pesan])


def tampilkan_dashboard_sidebar():
    """
    Baca log_aktivitas.csv dan tampilkan ringkasan statistik di sidebar.
    Dibungkus try-except karena file log mungkin belum ada sama sekali
    (aplikasi baru pertama kali dijalankan, belum ada aktivitas tercatat).
    """
    st.sidebar.title("📊 Dashboard Sistem")

    if not os.path.exists(LOG_PATH):
        st.sidebar.info("Belum ada aktivitas tercatat.")
        return

    df_log = pd.read_csv(LOG_PATH)

    total_aktivitas = len(df_log)
    n_info = (df_log['level'] == 'INFO').sum()
    n_warning = (df_log['level'] == 'WARNING').sum()
    n_error = (df_log['level'] == 'ERROR').sum()

    st.sidebar.metric("Total Aktivitas Tercatat", total_aktivitas)
    st.sidebar.metric("⚠️ Peringatan (Confidence Rendah)", n_warning)
    st.sidebar.metric("❌ Error", n_error)

    if n_error > 0:
        st.sidebar.markdown("**5 Error Terakhir:**")
        error_terakhir = df_log[df_log['level'] == 'ERROR'].tail(5)
        for _, row in error_terakhir.iterrows():
            st.sidebar.caption(f"🔴 {row['waktu']}: {row['pesan']}")


@st.cache_resource
def load_model_cached():
    from tensorflow.keras.models import load_model
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model AI (hanya sekali di awal)..."):
            gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)
    catat_log("INFO", "Model AI berhasil dimuat")
    return load_model(MODEL_PATH)


def klasifikasi_gambar(pil_image, model):
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if img_bgr.shape[0] < 10 or img_bgr.shape[1] < 10:
        raise ValueError(f"Ukuran gambar tidak wajar: {img_bgr.shape[:2]}")

    img_resized = cv2.resize(img_bgr, (64, 64))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    pred_probs = model.predict(img_input, verbose=0)[0]
    pred_idx = np.argmax(pred_probs)
    return {'label': CLASS_NAMES[pred_idx], 'confidence': float(pred_probs[pred_idx])}


# --- SIDEBAR (dijalankan di awal, sebelum konten utama) ---
tampilkan_dashboard_sidebar()

# --- UI UTAMA ---
st.title("🥕 Sistem Inspeksi Kualitas Bahan Makanan")
st.write("Selamat datang! Upload foto bahan makanan untuk memeriksa kesegarannya.")
st.markdown("---")

model = load_model_cached()

uploaded_file = st.file_uploader("Pilih foto bahan makanan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    catat_log("INFO", f"Foto diupload: {uploaded_file.name}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Foto Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Hasil Pemeriksaan")
        if st.button("🚀 Periksa Kualitas", type="primary"):
            try:
                hasil = klasifikasi_gambar(image, model)
                jenis_sayur = hasil['label'].split('_', 1)[1].capitalize()
                kondisi = "Segar ✅" if hasil['label'].startswith('fresh') else "Busuk ⚠️"
                confidence_pct = hasil['confidence'] * 100

                st.metric("Jenis Sayur", jenis_sayur)
                st.metric("Kondisi", kondisi)
                st.metric("Tingkat Keyakinan", f"{confidence_pct:.1f}%")

                if confidence_pct < 60:
                    st.warning("⚠️ Keyakinan model rendah — sebaiknya diverifikasi manual.")
                    catat_log("WARNING", f"Confidence rendah ({confidence_pct:.1f}%) untuk {uploaded_file.name}: {hasil['label']}")
                else:
                    catat_log("INFO", f"Klasifikasi berhasil: {uploaded_file.name} -> {hasil['label']} ({confidence_pct:.1f}%)")

            except Exception as e:
                st.error(f"❌ Gagal memproses gambar: {e}")
                catat_log("ERROR", f"Gagal memproses {uploaded_file.name}: {str(e)}")
        else:
            st.info("Klik tombol di atas untuk memulai pemeriksaan")
else:
    st.info("👆 Silakan upload foto terlebih dahulu")

st.markdown("---")
st.markdown("**Dibangun sebagai bagian dari Modul 12 — nalara.academy**")
