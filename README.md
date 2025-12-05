# 🏞️ Lake Segmentation Using K-Means Clustering

Proyek ini melakukan **segmentasi area danau** pada citra RGB menggunakan **K-Means clustering** berbasis warna dan **operasi morfologi** untuk mendapatkan pemisahan objek yang lebih bersih.  
Pendekatan ini berguna untuk pemetaan wilayah air, monitoring kualitas lingkungan, citra satelit, dan aplikasi computer vision lainnya.

---

## ✨ Fitur Utama
- Segmentasi citra berbasis **K-Means (4 cluster warna)**
- Deteksi otomatis area danau menggunakan **region terbawah gambar**
- **Morphological filtering** untuk memperhalus mask
- Visualisasi hasil lengkap:
  - Citra asli
  - Mask biner danau
  - Hasil segmentasi

---

## 📂 Struktur Kode
- **Input** : `lake.jpg` (citra target)
- **Output** : Mask & segmentasi area danau
- **Library** : `OpenCV`, `NumPy`, `Matplotlib`

---

## 🚀 Cara Menjalankan
install library yang diperlukan (wajib)
```bash
pip install opencv-python numpy matplotlib
```
jalankan script dengan:
```bash
python segment_lake.py
```
atau langsung run di Visual Studio Code

📄 Lisensi

MIT License
Created by Al Zakaria
