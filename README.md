# Lake Segmentation using K-Means Clustering

Sistem segmentasi danau otomatis menggunakan K-Means clustering dengan ekstraksi fitur multi-spektral untuk memisahkan area air dari vegetasi dan elemen lain dalam citra.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Deskripsi

Program ini mengimplementasikan metode segmentasi danau berbasis machine learning yang menggabungkan:
- **K-Means Clustering** untuk pengelompokan piksel
- **Multi-feature extraction** (HSV, LAB, RGB, texture, position)
- **Enhanced vegetation filtering** menggunakan ExG (Excess Green Index) dengan threshold adaptif
- **Spatial prior** untuk deteksi danau di posisi kiri atau bawah gambar
- **Morphological operations** untuk pembersihan mask

Sistem ini mampu membedakan area air (termasuk **danau hijau kebiruan**) dari vegetasi hijau dan elemen lanskap lainnya secara otomatis.

## ✨ Fitur Utama

- ✅ Ekstraksi fitur multi-spektral (11 channel)
- ✅ Deteksi otomatis cluster air dengan spatial prior (kiri & bawah)
- ✅ Pemisahan vegetasi dengan indeks ExG dan ExB
- ✅ Filtering tekstur untuk membedakan permukaan halus (air) dan kasar (vegetasi)
- ✅ **Support untuk danau hijau kebiruan** dengan threshold adaptif
- ✅ Seleksi komponen terbesar berdasarkan posisi (leftness) dan ukuran
- ✅ Morfologi adaptif untuk hasil yang halus

## 🛠️ Dependensi

```bash
opencv-python>=4.0.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## 📦 Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/lake-segmentation.git
cd lake-segmentation
```

2. Install dependensi:
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Cara Penggunaan

### Persiapan
Pastikan struktur folder sebagai berikut:
```
project/
├── process.py
├── sample/
│   └── yoursample.jpg          # Gambar input Anda
├── result/            # Otomatis dibuat
└── mask-clean/        # Dari fiji secara manual
```

### Menjalankan Program

```bash
python segment_lake.py
```

### Input
- Gambar input: `sample/yoursample.jpg`
- Format: JPG/PNG
- Resolusi: Fleksibel (tested pada 1000x1000 hingga 4000x3000)

### Output
Program akan menghasilkan 3 file:

1. **`mask-clean/lake_mask_binary.png`** - Mask biner area danau
2. **`mask-clean/vegetation_mask.png`** - Mask area vegetasi
3. **`result/lake_segmented.png`** - Hasil segmentasi final

## 🧮 Metodologi

### 1. Ekstraksi Fitur (11 Channel)
```
- L, a, b (LAB color space)
- H, S, V (HSV color space)
- ExG, ExB (Excess Green/Blue Index)
- Texture (Laplacian variance)
- Y-position (vertical location prior)
```

### 2. K-Means Clustering
```python
K = 5  # Jumlah cluster
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
```

### 3. Pemilihan Cluster Air
Heuristik scoring berdasarkan:
- **Tekstur rendah** (permukaan halus)
- **Saturasi rendah hingga sedang**
- **ExB positif** dan Lab b channel kebiruan
- **Penalti untuk EXG tinggi** (menghindari vegetasi)
- **Spatial prior**: Prioritas area di **kiri** atau **bawah** gambar
- **Penalti untuk value tinggi** (menghindari area terang seperti pasir)

### 4. Filtering Vegetasi (Enhanced)
```python
ExG > percentile(85)      # Lebih tinggi untuk melindungi danau hijau
Saturation > percentile(70)
Texture > percentile(65)  # Vegetasi lebih kasar
```

### 5. Morfologi & Komponen Terbesar
- Opening: Remove noise
- Closing: Fill holes
- Connected components: Select berdasarkan **size + leftness**
  - Prioritas komponen yang lebih besar DAN lebih kiri

## ⚙️ Konfigurasi

### Parameter Utama yang Dapat Disesuaikan

```python
# K-Means
K = 5  # Ubah ke 4-6 tergantung kompleksitas citra

# Bilateral Filter
d=7, sigmaColor=50, sigmaSpace=50

# Morfologi
kernel_size = (9, 9)  # Ubah untuk detil lebih halus/kasar

# Spatial Prior
bottom = slice(int(h*0.60), h)  # 60% bawah
left = slice(0, int(w*0.18))     # 18% kiri

# Threshold Vegetasi (percentile) - ENHANCED
exg_thr = np.percentile(exg, 85)   # 80-90 untuk proteksi danau hijau
sat_thr = np.percentile(S, 70)     # 65-75
tex_thr = np.percentile(texture_norm, 65)  # 60-70
```

### Tuning untuk Kasus Berbeda

| Kondisi | Solusi |
|---------|--------|
| Air terlalu gelap | Turunkan `val_med` penalty weight |
| Banyak vegetasi hijau muda | Naikkan `sat_thr` percentile ke 75+ |
| **Danau hijau kebiruan** | Naikkan `exg_thr` ke 85-90 |
| Air dengan refleksi | Turunkan `tex_thr` percentile |
| Danau kecil tidak terdeteksi | Turunkan area threshold di connected components |
| **Danau di posisi kiri** | Pastikan `left_frac` weight cukup tinggi (1.6+) |
| Vegetasi ikut terdeteksi | Naikkan `exg_med` penalty weight |

## 📊 Contoh Hasil

### Input
<p align="center">Citra danau dengan vegetasi di sekitarnya</p>

### Output
<table>
  <tr>
    <td><b>Mask Vegetasi</b></td>
    <td><b>Mask Danau</b></td>
    <td><b>Segmentasi Final</b></td>
  </tr>
  <tr>
    <td>Filter area hijau</td>
    <td>Deteksi area air</td>
    <td>Hasil akhir</td>
  </tr>
</table>

## 🔬 Kasus Penggunaan

- Monitoring ekologi danau dan waduk
- Pemetaan sumber daya air
- Analisis perubahan luas permukaan air
- Deteksi kualitas air berdasarkan warna (hijau kebiruan)
- Segmentasi danau dengan posisi tidak standar (kiri/kanan gambar)
- Dataset preparation untuk deep learning
- Analisis citra satelit atau drone

## 🐛 Troubleshooting

### Error: "Gambar tidak ditemukan"
- Pastikan file `sample/2.jpg` ada
- Periksa path dan nama file

### Mask terlalu banyak noise
- Tingkatkan iterations pada morphologyEx
- Perbesar kernel size

### Vegetasi ikut terdeteksi
- Naikkan `exg_thr` percentile ke 85-90
- Naikkan `tex_thr` percentile ke 65-70
- Tingkatkan penalty weight untuk `exg_med`

### Danau hijau terdeteksi sebagai vegetasi
- **Ini adalah improvement utama versi ini!**
- `exg_thr` sudah dinaikkan ke percentile 85
- `tex_thr` sudah dinaikkan ke 65 (vegetasi lebih kasar)
- Jika masih masalah, coba naikkan lagi ke 90

### Air tidak terdeteksi sama sekali
- Turunkan K (coba K=4)
- Kurangi weight texture pada scoring
- Periksa apakah image terlalu gelap

## 📝 Lisensi

MIT License

Copyright (c) 2025 Al Zakaria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 👤 Author

**Al Zakaria**

Jika proyek ini bermanfaat, jangan lupa berikan ⭐ di GitHub!

## 🤝 Kontribusi

Kontribusi, issues, dan feature requests sangat diterima!

1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📚 Referensi

- [OpenCV Documentation](https://docs.opencv.org/)
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Excess Green Index for Vegetation](https://doi.org/10.13031/2013.27838)
- [LAB Color Space](https://en.wikipedia.org/wiki/CIELAB_color_space)

## 📧 Kontak

Untuk pertanyaan atau diskusi, silakan buat issue di repository ini.

---

<p align="center">Made with ❤️ by Al Zakaria</p>
