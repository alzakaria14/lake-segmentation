import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# 1. BACA GAMBAR
# ======================================
img = cv2.imread('lake.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# ======================================
# 2. K-MEANS CLUSTERING (3–4 KLASTER)
# ======================================
Z = img_rgb.reshape((-1, 3))
Z = np.float32(Z)

K = 4   # jumlah cluster
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

label_image = label.reshape((h, w))

# ======================================
# 3. DETEKSI KLASTER YANG DOMINAN DI BAGIAN BAWAH
# ======================================
bottom_region = label_image[int(h*0.60):, :]   # 40% paling bawah
unique, counts = np.unique(bottom_region, return_counts=True)

# cluster yang area-nya paling besar → cluster danau
lake_cluster = unique[np.argmax(counts)]

# buat mask danau
mask_lake = (label_image == lake_cluster).astype(np.uint8) * 255

# ======================================
# 4. PERHALUS MASK DENGAN MORFOLOGI
# ======================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_lake_clean = cv2.morphologyEx(mask_lake, cv2.MORPH_OPEN, kernel, iterations=2)
mask_lake_clean = cv2.morphologyEx(mask_lake_clean, cv2.MORPH_CLOSE, kernel, iterations=3)

# ======================================
# 5. TERAPKAN MASK
# ======================================
lake_segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_lake_clean)

# ======================================
# 6. TAMPILKAN HASIL
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
