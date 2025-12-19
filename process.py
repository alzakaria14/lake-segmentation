import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================
# 0. BUAT FOLDER OUTPUT
# ======================================
os.makedirs("result", exist_ok=True)
os.makedirs("mask-clean", exist_ok=True)

# ======================================
# 1. BACA GAMBAR
# ======================================
img_bgr = cv2.imread("sample/5.jpg")
if img_bgr is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Cek path: sample/5.jpg")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# (opsional) peredam noise biar fitur tekstur lebih stabil
img_rgb_smooth = cv2.bilateralFilter(img_rgb, d=7, sigmaColor=50, sigmaSpace=50)

# ======================================
# 2. EKSTRAK FITUR (AMAN DARI RESHAPE ERROR)
# ======================================
hsv = cv2.cvtColor(img_rgb_smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
lab = cv2.cvtColor(img_rgb_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)

H = hsv[:, :, 0] / 179.0
S = hsv[:, :, 1] / 255.0
V = hsv[:, :, 2] / 255.0

L = lab[:, :, 0] / 255.0
a = (lab[:, :, 1] - 128.0) / 128.0
b = (lab[:, :, 2] - 128.0) / 128.0

R = img_rgb_smooth[:, :, 0] / 255.0
G = img_rgb_smooth[:, :, 1] / 255.0
B = img_rgb_smooth[:, :, 2] / 255.0

exg = 2*G - R - B
exb = 2*B - R - G

gray = cv2.cvtColor(img_rgb_smooth, cv2.COLOR_RGB2GRAY)
lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
texture = cv2.GaussianBlur(np.abs(lap), (0, 0), 1.0)
texture_norm = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)

# POSISI VERTIKAL (AMAN)
ypos = np.tile(
    np.linspace(0, 1, h, dtype=np.float32)[:, None],
    (1, w)
)

# STACK FITUR → (h, w, 11)
features = np.dstack([
    L, a, b,
    H, S, V,
    exg, exb,
    texture_norm,
    ypos
])

# RESHAPE → (h*w, 11)
features = features.reshape((-1, features.shape[-1])).astype(np.float32)


# ======================================
# 3. K-MEANS CLUSTERING PADA FITUR
# ======================================
K = 5  # biasanya 4-6 lebih stabil dari 3-4 untuk kasus hijau vs hijau
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
_, labels, centers = cv2.kmeans(
    features, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
)
label_image = labels.reshape((h, w))

# ======================================
# 4. PILIH CLUSTER "AIR" (LEBIH ROBUST)
#    Air: tekstur rendah, tidak terlalu hijau-vegetasi (EXG tidak ekstrem),
#         cenderung dominan di kiri ATAU bawah (bukan hanya bawah).
# ======================================
best_c = None
best_score = -1e9

bottom = slice(int(h*0.60), h)
left = slice(0, int(w*0.18))   # 18% kiri

for c in range(K):
    m = (label_image == c)
    if m.sum() < 0.01 * h * w:
        continue

    sat_med = np.median(S[m])
    val_med = np.median(V[m])
    tex_med = np.median(texture_norm[m])
    exg_med = np.median(exg[m])
    exb_med = np.median(exb[m])
    b_med   = np.median(b[m])  # Lab b*

    bottom_frac = m[bottom, :].mean()
    left_frac   = m[:, left].mean()

    # Skor utama: halus + tidak terlalu saturated + tidak EXG ekstrem (vegetasi)
    score = 0
    score += 2.0 * (1.0 - tex_med)
    score += 0.8 * (1.0 - sat_med)

    # Air biru: exb tinggi, b* lebih "dingin"
    score += 0.7 * exb_med
    score += 0.3 * (-b_med)

    # Vegetasi biasanya EXG besar positif → kasih penalti, tapi jangan terlalu keras
    score -= 0.35 * max(0.0, exg_med)

    # Prior lokasi: di sampel Anda danau dominan kiri, bukan hanya bawah
    score += 1.6 * max(left_frac, bottom_frac)

    # Hindari cluster yang terlalu terang (sering darat/pasir)
    score -= 0.35 * max(0.0, val_med - 0.75)

    if score > best_score:
        best_score = score
        best_c = c

if best_c is None:
    raise RuntimeError("Gagal memilih cluster air. Coba naikkan K (mis. 6).")

mask_water = (label_image == best_c).astype(np.uint8) * 255

# ======================================
# 5. KURANGI VEGETASI (LEBIH AMAN UNTUK DANAU HIJAU)
#    Vegetasi: EXG tinggi + tekstur tinggi + saturasi tinggi.
#    Air hijau biasanya lebih halus (tekstur rendah), jadi tidak ikut terhapus.
# ======================================
exg_thr = np.percentile(exg, 85)          # lebih tinggi → tidak gampang menganggap danau hijau sebagai vegetasi
sat_thr = np.percentile(S, 70)
tex_thr = np.percentile(texture_norm, 65) # vegetasi cenderung lebih "kasar"

veg_mask = ((exg > exg_thr) & (S > sat_thr) & (texture_norm > tex_thr)).astype(np.uint8) * 255
mask_water = cv2.bitwise_and(mask_water, cv2.bitwise_not(veg_mask))

# hapus vegetasi dari kandidat air
mask_water = cv2.bitwise_and(mask_water, cv2.bitwise_not(veg_mask))

# ======================================
# 6. PERHALUS MASK (MORFOLOGI + PILIH KOMPONEN TERBESAR DEKAT BAWAH)
# ======================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_clean = cv2.morphologyEx(mask_water, cv2.MORPH_OPEN, kernel, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=3)

# pilih komponen yang paling "danau": besar dan dekat bawah
num, cc, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
if num > 1:
    best_id = 0
    best_cc_score = -1e9

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]  # posisi kiri bbox
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        # komponen yang lebih besar dan lebih dekat sisi kiri dapat skor lebih tinggi
        leftness = 1.0 - (x / (w + 1e-6))   # makin kiri makin besar
        size = area / (h*w)

        s = 1.3 * size + 0.9 * leftness
        if s > best_cc_score:
            best_cc_score = s
            best_id = i

    mask_clean = (cc == best_id).astype(np.uint8) * 255


# ======================================
# 7. TERAPKAN MASK & SIMPAN
# ======================================
lake_segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)

cv2.imwrite("mask-clean/lake_mask_binary.png", mask_clean)
cv2.imwrite("mask-clean/vegetation_mask.png", veg_mask)
cv2.imwrite("result/lake_segmented.png", cv2.cvtColor(lake_segmented, cv2.COLOR_RGB2BGR))

print("Mask danau disimpan: mask-clean/lake_mask_binary.png")
print("Mask vegetasi disimpan: mask-clean/vegetation_mask.png")
print("Hasil segmentasi disimpan: result/lake_segmented.png")

# ======================================
# 8. TAMPILKAN
# ======================================
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.title("Citra Asli")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Mask Vegetasi (perkiraan)")
plt.imshow(veg_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Mask Danau (final)")
plt.imshow(mask_clean, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Segmentasi Danau")
plt.imshow(lake_segmented)
plt.axis("off")

plt.tight_layout()
plt.show()
