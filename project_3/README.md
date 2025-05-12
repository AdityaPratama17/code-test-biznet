# Sistem Deteksi Kecurangan (Fraud) Reimbursement dengan Machine Learning

Sistem ini dirancang untuk mengidentifikasi aktivitas mencurigakan dalam proses reimbursement perusahaan menggunakan pendekatan Machine Learning (ML) dan Artificial Intelligence (AI). Terdapat tiga modul utama:


## 1. Fraud Perjalanan Bisnis
**Data yang Digunakan:**
- `tanggal`: Tanggal transaksi.
- `jarak` (km): Jarak dari lokasi asal ke tujuan.
- `tipe_transportasi`: Jenis transportasi (mobil, pesawat, kereta, dll.).
- `biaya`: Nominal pengajuan reimbursement.
- `is_fraud` (label): Label apakah transaksi dicurigai fraud (1) atau valid (0).

**Pendekatan ML:**
- Algoritma Klasifikasi seperti Random Forest, XGBoost, atau Logistic Regression untuk memprediksi kemungkinan fraud berdasarkan pola historis.
- Anomali Detection seperti Isolation Forest untuk menandai pengajuan dengan biaya/jarak tidak wajar.


## 2. Fraud Nota/Bukti Transfer
**Alur Deteksi:**

1. **OCR (Optical Character Recognition)**
    - Ekstrak teks dari gambar nota/struk/bukti transfer menggunakan tools seperti Tesseract (Python) atau Google Vision API.

2. **Parsing dengan LLM (Large Language Model)**
    - Gunakan model LLM untuk memahami konteks teks (misal: nama merchant, nominal, tanggal).

3. **Pencocokan Data**
    - Bandingkan hasil parsing dengan data yang diinput pengaju. Jika ada perbedaan signifikan (misal: nominal tidak sesuai), sistem memberi flag fraud.


## 3. Fraud Transaksi Kartu Kredit Perusahaan
**Data yang Digunakan:**
- `pegawai`: Nama pegawai pemegang kartu.
- `tanggal`: Tanggal transaksi.
- `tipe`: Kategori pengeluaran (makan, akomodasi, belanja, dll.).
- `old_balance`: Saldo sebelum transaksi.
- `new_balance`: Saldo setelah transaksi.
- `biaya`: Nominal transaksi.
- `is_fraud` (label): Label fraud (1/0).

**Pendekatan ML:**
- Algoritma Klasifikasi seperti Random Forest, XGBoost, atau Logistic Regression untuk memprediksi kemungkinan transaksi mencurigakan berdasarkan pola historis.


# Teknologi yang Digunakan
- Bahasa Pemrograman: Python.
- ML Framework: Scikit-learn.
- OCR: Tesseract.
- LLM: GPT-4, DeepSeek.
- Visualisasi: Matplotlib, Seaborn.

