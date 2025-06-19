import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model

# Judul aplikasi
st.title("Klasifikasi Larva BSF vs Lalat Hijau")
st.write("Unggah gambar larva untuk mengidentifikasi apakah itu larva Black Soldier Fly (BSF) atau larva lalat hijau")

# Load model yang sudah disimpan
@st.cache_resource
def load_my_model():
    model = load_model('best_cnn_model.h5') 
    return model

model = load_my_model()

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_larva(img, model):
    # Resize gambar ke ukuran yang diharapkan model (misal 224x224)
    img = img.resize((224, 224))
    
    # Konversi gambar ke array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    
    # Lakukan prediksi
    prediction = model.predict(img_array)
    
    return prediction

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar larva...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Gambar Larva yang Diunggah', use_column_width=True)
    
    # Lakukan prediksi ketika tombol ditekan
    if st.button('Identifikasi Larva'):
        with st.spinner('Sedang memproses gambar...'):
            # Proses gambar dan prediksi
            prediction = predict_larva(image_display, model)
            
            # Ambil hasil prediksi
            class_names = ['Larva BSF', 'Larva Lalat Hijau']  # Sesuaikan dengan urutan kelas di model Anda
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Tampilkan hasil
            st.success(f"Hasil Identifikasi: {predicted_class}")
            st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
            
            # Tambahkan penjelasan
            if predicted_class == "Larva BSF":
                st.info("""
                **Ciri-ciri Larva BSF:**
                - Warna krem atau kecoklatan
                - Bentuk tubuh meruncing di kedua ujung
                - Ukuran lebih besar dan gemuk
                - Bergerak aktif tapi tidak terlalu cepat
                """)
            else:
                st.info("""
                **Ciri-ciri Larva Lalat Hijau:**
                - Warna putih atau kekuningan
                - Tubuh lebih ramping dan panjang
                - Bergerak sangat aktif
                - Sering ditemukan pada bahan organik yang membusuk
                """)

# Catatan kaki
st.markdown("---")
st.caption("Aplikasi ini menggunakan model deep learning untuk mengidentifikasi jenis larva. Pastikan gambar yang diunggah jelas dan fokus pada larva.")