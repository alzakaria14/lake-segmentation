import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os # Added os module import

def extract_color_features(image):
    """
    Mengekstraksi statistik warna (Rata-rata dan Standar Deviasi)
    dari ruang warna HSV. HSV lebih tahan terhadap perubahan cahaya
    dibanding RGB.
    """
    # Konversi ke HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Pisahkan channel H, S, dan V
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # Hitung Mean (Rata-rata) dan Std Dev (Simpangan Baku)
    # Ini memberi kita 6 fitur warna
    color_features = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]
    return color_features

def extract_texture_features(image):
    """
    Mengekstraksi tekstur duri menggunakan GLCM (Gray-Level Co-occurrence Matrix).
    Fitur ini sangat bagus untuk membedakan pola duri yang kasar vs halus.
    """
    # Konversi ke Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menghitung GLCM
    # distances=[1]: membandingkan piksel dengan tetangga langsungnya
    # angles: 0, 45, 90, 135 derajat (untuk rotasi invariant)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    # Properti yang diekstrak
    # 1. Contrast: Mengukur perbedaan intensitas lokal (ketajaman duri)
    # 2. Dissimilarity: Kemiripan antar piksel
    # 3. Homogeneity: Kehalusan tekstur
    # 4. Energy: Keseragaman tekstur
    # 5. Correlation: Ketergantungan linear antar piksel

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    texture_features = []
    for prop in properties:
        # Ambil rata-rata dari 4 sudut agar fitur tidak terpengaruh rotasi gambar
        val = graycoprops(glcm, prop)
        texture_features.append(np.mean(val))

    return texture_features

def process_durian_image(image_path):
    # 1. Baca Gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Gambar {image_path} tidak ditemukan")
        return None

    # OPSIONAL: Resize gambar agar proses komputasi lebih cepat & seragam
    img = cv2.resize(img, (256, 256))

    # 2. Ekstraksi Fitur
    print(f"Sedang memproses: {image_path}...")

    # Ambil ciri warna (6 fitur)
    f_color = extract_color_features(img)

    # Ambil ciri tekstur (5 fitur)
    f_texture = extract_texture_features(img)

    # 3. Gabungkan menjadi satu vektor fitur (Feature Vector)
    # Total 11 fitur yang siap masuk ke Klasifikasi (SVM/KNN/Neural Network)
    feature_vector = f_color + f_texture

    return feature_vector

# --- CONTOH PENGGUNAAN ---

# Ganti dengan nama file gambar durian Anda
# Misalnya: 'durian_montong.jpg'
file_name = 'lake-forest-crop.jpg'

# Buat file dummy jika tidak ada (Hanya untuk demo agar kode tidak error saat di-copy)
#if not os.path.exists(file_name): # Corrected line: removed 'import' keyword
#    dummy_img = np.zeros((256,256,3), dtype=np.uint8)
#    cv2.imwrite(file_name, dummy_img)

# Jalankan Ekstraksi
features = process_durian_image(file_name)

if features:
    print("\n--- HASIL EKSTRAKSI CIRI ---")
    labels = [
        "H_Mean", "H_Std", "S_Mean", "S_Std", "V_Mean", "V_Std",  # Warna
        "Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation" # Tekstur
    ]

    for label, value in zip(labels, features):
        print(f"{label:<15}: {value:.4f}")

    print(f"\nTotal Fitur: {len(features)}")
    print("Vektor ini siap digunakan untuk training model (SVM/KNN).")