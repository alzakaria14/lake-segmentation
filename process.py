import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================
# 0. BUAT FOLDER OUTPUT JIKA BELUM ADA
# ======================================
os.makedirs("result", exist_ok=True)
os.makedirs("mask-clean", exist_ok=True)

# ======================================
# 1. BACA GAMBAR
# ======================================
img = cv2.imread('sample/1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# ======================================
# 2. K-MEANS CLUSTERING (3–4 KLASTER)
# ======================================
Z = img_rgb.reshape((-1, 3))
Z = np.float32(Z)

K = 4   # jumlah cluster
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
ret, label, center = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

label_image = label.reshape((h, w))

# ======================================
# 3. DETEKSI KLASTER DOMINAN DI BAGIAN BAWAH
# ======================================
bottom_region = label_image[int(h * 0.60):, :]
unique, counts = np.unique(bottom_region, return_counts=True)

lake_cluster = unique[np.argmax(counts)]

mask_lake = (label_image == lake_cluster).astype(np.uint8) * 255

# ======================================
# 4. PERHALUS MASK DENGAN MORFOLOGI
# ======================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_lake_clean = cv2.morphologyEx(
    mask_lake, cv2.MORPH_OPEN, kernel, iterations=2
)
mask_lake_clean = cv2.morphologyEx(
    mask_lake_clean, cv2.MORPH_CLOSE, kernel, iterations=3
)

# ======================================
# 5. TERAPKAN MASK
# ======================================
lake_segmented = cv2.bitwise_and(
    img_rgb, img_rgb, mask=mask_lake_clean
)

# ======================================
# 6. SIMPAN FILE HASIL
# ======================================
cv2.imwrite("mask-clean/lake_mask_binary.png", mask_lake_clean)
cv2.imwrite(
    "result/lake_segmented.png",
    cv2.cvtColor(lake_segmented, cv2.COLOR_RGB2BGR)
)

print("Mask disimpan di folder mask-clean/")
print("Hasil segmentasi disimpan di folder result/")

# ======================================
# 7. TAMPILKAN HASIL
# ======================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title('Citra Asli')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Mask Danau (b/w)')
plt.imshow(mask_lake_clean, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Segmentasi Danau')
plt.imshow(lake_segmented)
plt.axis('off')

plt.tight_layout()
plt.show()
