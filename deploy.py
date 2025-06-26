# -*- coding: utf-8 -*-
"""deploy.py"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops

model = load_model('best_cnn_model.h5')

label_map = {0: 'BSF', 1: 'HouseFly'}

st.title('Klasifikasi Larva BSF vs Lalat Rumah')

uploaded = st.file_uploader('Unggah gambar larva', type=['jpg', 'jpeg', 'png'])

st.image(uploaded, caption='Gambar yang diunggah', use_column_width=True)

img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
denoised = cv2.fastNlMeansDenoising(clahe, h=7, templateWindowSize=13)
gaussian = cv2.GaussianBlur(denoised, (0, 0), 0.1)
sharpened = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)
thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)

mask = np.zeros(sharpened.shape, np.uint8)
mask[thresh == 255] = cv2.GC_PR_FGD
mask[thresh == 0] = cv2.GC_PR_BGD
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(rgb, mask, None, bgdModel, fgdModel, 25, cv2.GC_INIT_WITH_MASK)
final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
masked_img = sharpened * final_mask

coords = np.argwhere(final_mask)
x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1
cropped = masked_img[x0:x1, y0:y1]

glcm = graycomatrix(cropped, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
features = [
    graycoprops(glcm, 'contrast').mean(),
    graycoprops(glcm, 'dissimilarity').mean(),
    graycoprops(glcm, 'homogeneity').mean(),
    graycoprops(glcm, 'energy').mean(),
    graycoprops(glcm, 'correlation').mean(),
    graycoprops(glcm, 'ASM').mean()
]

x_pred = np.array(features).reshape(1, 6, 1)

if st.button('Prediksi'):
    prediction = model.predict(x_pred)
    predicted_index = np.argmax(prediction)
    predicted_label = label_map[predicted_index]
    st.success(f'Hasil Prediksi: {predicted_label}')
