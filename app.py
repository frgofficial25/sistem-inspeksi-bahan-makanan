
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import gdown
import csv
import pandas as pd
import re
import plotly.graph_objects as go
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
WARNA_LEVEL = {'INFO': '#4CAF50', 'WARNING': '#FFC107', 'ERROR': '#F44336'}


def catat_log(level, pesan):
    file_baru = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if file_baru:
            writer.writerow(['waktu', 'level', 'pesan'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level, pesan])


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


def ekstrak_kelas(pesan):
    match = re.search(r'-> (\w+)', pesan)
    if match:
        return match.group(1)
    match2 = re.search(r': (\w+_\w+)', pesan)
    if match2:
        return match2.group(1)
    return None


def tampilkan_dashboard_sidebar():
    st.sidebar.title("📊 Dashboard Sistem")
    if not os.path.exists(LOG_PATH):
        st.sidebar.info("Belum ada aktivitas tercatat.")
        return
    df_log = pd.read_csv(LOG_PATH)
    st.sidebar.metric("Total Aktivitas Tercatat", len(df_log))
    st.sidebar.metric("⚠️ Peringatan (Confidence Rendah)", (df_log['level'] == 'WARNING').sum())
    st.sidebar.metric("❌ Error", (df_log['level'] == 'ERROR').sum())


def halaman_periksa_kualitas(model):
    st.title("🥕 Sistem Inspeksi Kualitas Bahan Makanan")
    st.write("Upload foto bahan makanan untuk memeriksa kesegarannya.")
    st.markdown("---")

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


def halaman_dashboard():
    st.title("📊 Executive Dashboard")
    st.write("Ringkasan aktivitas sistem inspeksi kualitas bahan makanan.")
    st.markdown("---")

    if not os.path.exists(LOG_PATH):
        st.info("Belum ada data aktivitas untuk ditampilkan. Coba periksa beberapa foto dulu di tab 'Periksa Kualitas'.")
        return

    df_log = pd.read_csv(LOG_PATH)
    df_log['waktu'] = pd.to_datetime(df_log['waktu'])
    df_log['tanggal'] = df_log['waktu'].dt.date
    df_log['kelas_terdeteksi'] = df_log['pesan'].apply(ekstrak_kelas)

    # Metrik ringkas di atas
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Aktivitas", len(df_log))
    col2.metric("Peringatan", (df_log['level'] == 'WARNING').sum())
    col3.metric("Error", (df_log['level'] == 'ERROR').sum())

    st.markdown("---")

    # Chart tren (kalau data cukup, ada variasi tanggal)
    if df_log['tanggal'].nunique() > 1:
        agregasi_harian = df_log.groupby(['tanggal', 'level']).size().unstack(fill_value=0)
        fig_tren = go.Figure()
        for level in ['INFO', 'WARNING', 'ERROR']:
            if level in agregasi_harian.columns:
                fig_tren.add_trace(go.Scatter(
                    x=agregasi_harian.index.astype(str), y=agregasi_harian[level],
                    mode='lines+markers', name=level,
                    line=dict(color=WARNA_LEVEL[level], width=3)
                ))
        fig_tren.update_layout(title='Tren Aktivitas Harian', template='plotly_dark', height=350)
        st.plotly_chart(fig_tren, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        level_counts = df_log['level'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=level_counts.index, values=level_counts.values,
            marker=dict(colors=[WARNA_LEVEL[l] for l in level_counts.index]), hole=0.4
        )])
        fig_pie.update_layout(title='Proporsi Status', template='plotly_dark', height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        kelas_counts = df_log['kelas_terdeteksi'].dropna().value_counts().head(10)
        if not kelas_counts.empty:
            fig_bar = go.Figure(data=[go.Bar(
                x=kelas_counts.values, y=kelas_counts.index.str.replace('_', ' ').str.title(),
                orientation='h', marker=dict(color='#4CAF50')
            )])
            fig_bar.update_layout(title='Bahan Makanan Terlaris', template='plotly_dark',
                                    height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)


# --- SIDEBAR & NAVIGASI UTAMA ---
tampilkan_dashboard_sidebar()
model = load_model_cached()

tab1, tab2 = st.tabs(["🔍 Periksa Kualitas", "📊 Dashboard"])

with tab1:
    halaman_periksa_kualitas(model)

with tab2:
    halaman_dashboard()

st.markdown("---")
st.markdown("**Dibangun sebagai bagian dari Modul 13 — nalara.academy**")
