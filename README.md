# Semantic Search untuk Merchant
### Search & Discovery berbasis Embedding, Olist Brazilian E-Commerce Dataset

> Proyek ini membangun dua mesin pencari untuk katalog merchant e-commerce, lalu membandingkan keduanya: pencarian kata kunci (TF-IDF) dan pencarian makna (semantic search berbasis embedding). Dataset Olist tidak punya deskripsi merchant, jadi langkah pertama proyek ini menyusun ulang profil teks tiap seller dari data transaksi. Hasilnya: perbandingan dua engine di 110 query, plus fitur personalisasi yang menyesuaikan hasil dengan histori belanja pelanggan.

## 1. Latar Belakang & Business Questions

Pencarian kata kunci cuma cocok kalau kata di query sama persis dengan kata di teks yang dicari. Query "perlengkapan bayi murah" bisa gagal menemukan merchant yang relevan kalau profilnya ditulis dengan kata lain. Semantic search mengatasi ini dengan mencocokkan makna, bukan kata per kata.

Tiga pertanyaan yang dijawab proyek ini:

1. **Seberapa sering pengguna gagal menemukan merchant yang mereka cari lewat pencarian saat ini?**
   *Diukur pakai: Recall@10 dan studi kasus kualitatif pada TF-IDF.*
2. **Kalau mesin pencari diganti ke yang berbasis makna (bukan kata kunci), apakah pengguna lebih sering menemukan merchant yang relevan?**
   *Diukur pakai: Recall@10, Precision@10, MRR  TF-IDF vs semantic.*
3. **Bisakah urutan hasil pencarian disesuaikan dengan kebiasaan belanja tiap pelanggan, tanpa membuat hasilnya kurang relevan?**
   *Diukur pakai: overlap top-5 vs skor relevansi query, di berbagai nilai `alpha`, untuk satu query uji ("produk elektronik") pada engine semantic.  

## 2. Sumber Data

[Olist Brazilian E-Commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 9 tabel CSV (customers, orders, order_items, order_payments, order_reviews, products, sellers, geolocation, category_translation). Data seller cuma berisi `seller_id`, kode pos, kota, dan provinsi  tidak ada nama atau deskripsi toko. Karena itu, profil teks tiap merchant harus disusun dari riwayat transaksinya, bukan diambil langsung dari data.

## 3. Metodologi / Alur Kerja

| Tahap | Isi | Output kunci |
|---|---|---|
| 1. Setup & load data | Baca 9 CSV Olist | 9 DataFrame mentah |
| 2. Tantangan data | Cek struktur `sellers`: tidak ada nama/deskripsi | Justifikasi rekonstruksi profil |
| 3. Bangun profil teks merchant | Gabungkan order_items → products → kategori (EN→ID) → orders → reviews → sellers, lalu susun jadi kalimat per seller (min. 3 order) | 2.138 profil merchant siap diembed |
| 4. Baseline TF-IDF | `TfidfVectorizer` (unigram+bigram) + cosine similarity | Engine TF-IDF, vocab 5.281 term |
| 5. Semantic search | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) + FAISS `IndexFlatIP`, fallback otomatis ke TF-IDF+SVD kalau tidak ada internet | Engine semantic |
| 6. Bandingkan engine | 3 query bahasa sehari-hari dicoba di kedua engine | Studi kasus kualitatif |
| 7. Evaluasi kuantitatif | Ground truth sintetis dari taksonomi kategori (110 query), hitung Recall@10 / Precision@10 / MRR | Tabel & chart perbandingan |
| 8. Personalisasi | Gabungkan skor query (engine semantic) dengan skor histori kategori pelanggan (≥2 order), untuk 1 query uji ("produk elektronik") | Evaluasi trade-off relevansi vs personalisasi |

## 4. Tech Stack

- Python: pandas, numpy
- Search: scikit-learn (`TfidfVectorizer`, cosine similarity), `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), FAISS (`IndexFlatIP`)
- Evaluasi: Recall@K, Precision@K, MRR, dihitung manual dari ground truth sintetis
- Visualisasi: matplotlib
- Demo terpisah: Streamlit (`app.py`, di luar notebook ini)

## 5. Cara Kerja Tiap Algoritma

**TF-IDF.** Cara kerjanya: kata yang sering muncul di satu profil merchant tapi jarang muncul di profil lain dianggap penting untuk profil itu. Kata umum yang muncul di hampir semua profil (seperti "produk" atau "bagus") dianggap kurang penting.

```
tfidf(t, d) = tf(t, d) × log(N / df(t))
```

`tf(t, d)`: berapa kali kata `t` muncul di dokumen `d`. `N`: jumlah total dokumen. `df(t)`: di berapa dokumen kata itu muncul. Query dan profil merchant diubah jadi angka pakai rumus ini, lalu dibandingkan pakai cosine similarity.

**Cosine similarity.** Ini rumus yang dipakai TF-IDF dan semantic search untuk mengukur "seberapa mirip" dua teks setelah diubah jadi angka (vektor):

```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Intinya: mengukur arah, bukan panjang. Teks panjang dan teks pendek dengan isi mirip tetap dianggap mirip, karena yang dibandingkan cuma arahnya.

**Semantic search.** Query dan profil merchant diubah jadi angka pakai model AI (`paraphrase-multilingual-MiniLM-L12-v2`), bukan dihitung dari kemunculan kata seperti TF-IDF. Model ini sudah dilatih supaya kalimat dengan arti mirip menghasilkan angka yang berdekatan, walau kata-katanya beda. FAISS (`IndexFlatIP`) dipakai untuk mencari kecocokan dengan cepat  hasilnya setara dengan cosine similarity, tapi jauh lebih efisien kalau datanya besar.

**Personalisasi.** Skor akhir tiap merchant adalah campuran dua hal: relevansi terhadap query, dan kecocokan dengan kebiasaan belanja pelanggan.

```
final_score = alpha × score_query + (1 - alpha) × score_history
```

`alpha` mengatur porsi campurannya. `alpha = 1.0` berarti murni relevansi query, tanpa personalisasi. Makin kecil `alpha`, makin besar pengaruh histori pelanggan.

## 6. Cara Mengukur Hasilnya

Tiga metrik ini dihitung dari 110 query ground truth sintetis:

**Recall@K**  dari semua merchant yang seharusnya muncul, berapa persen yang benar-benar masuk ke K hasil teratas. Recall rendah berarti banyak merchant relevan yang terlewat.

```
Recall@K = (merchant relevan yang masuk top-K) / (total merchant relevan)
```

**Precision@K**  dari K hasil yang ditampilkan, berapa persen yang benar-benar relevan. Precision rendah berarti banyak hasil yang "nyasar".

```
Precision@K = (merchant relevan yang masuk top-K) / K
```

**MRR (Mean Reciprocal Rank)**  seberapa cepat hasil yang relevan muncul, dirata-rata dari semua query. MRR tinggi berarti hasil paling relevan biasanya muncul di urutan atas, bukan terkubur di posisi bawah.

```
MRR = (1 / |Q|) × Σ (1 / rank_i)
```

`|Q|`: jumlah query. `rank_i`: posisi hasil relevan pertama untuk query ke-`i`.

## 7. Hasil & Insight Kunci

### 7.1 Business Question 1  seberapa sering pengguna gagal menemukan merchant yang mereka cari?

TF-IDF melewatkan rata-rata ~42% merchant relevan di top-10 (Recall@10 = 0,579). Kegagalannya bukan cuma "tidak ketemu"  kadang TF-IDF malah salah arah kalau ada kata yang kebetulan sama tapi maksudnya beda. Contoh: query "biar kerja dari rumah makin nyaman" nyasar ke kategori "material konstruksi rumah", gara-gara kata "rumah".

### 7.2 Business Question 2  apakah mesin pencari berbasis makna membantu pengguna lebih baik?

![Perbandingan TF-IDF vs Semantic](assets/engine_comparison.png)

| Engine | Recall@10 | Precision@10 | MRR |
|---|---|---|---|
| TF-IDF | 0,579 | 0,510 | 0,777 |
| Semantic | 0,604 | 0,525 | 0,773 |

Semantic search menang di recall dan precision (naik 2,5 dan 1,5 poin), tapi MRR-nya sedikit kalah  TF-IDF masih sedikit lebih jago menaruh hasil terbaik di posisi #1. Selisihnya nyata tapi tidak besar, karena ground truth dibuat dari kata kunci kategori yang sama persis dengan teks profil merchant, jadi menguntungkan TF-IDF yang mengandalkan kecocokan kata.

Semantic search juga tidak selalu menang. Performanya tergantung seberapa dekat kata-kata di query dengan kosakata yang ada di profil merchant.

### 7.3 Business Question 3  bisakah hasil disesuaikan per pelanggan tanpa merusak relevansi?

**Cakupan pengujian:** trade-off ini diukur pada 50 pelanggan (yang punya histori ≥2 order) dan 4 nilai `alpha`, tapi seluruhnya untuk **satu query uji** ("produk elektronik") dan **satu engine** (semantic). Ini cukup untuk melihat pola trade-off secara terukur (bukan cuma 1 contoh anekdotal seperti demo awal), tapi belum berarti pola ini pasti sama untuk query lain atau untuk TF-IDF  lihat Batasan poin 4.

![Trade-off personalisasi](assets/personalization_tradeoff.png)

| alpha | Overlap top-5 vs non-personalized | Relevansi query |
|---|---|---|
| 1,0 | 100% | 0,928 |
| 0,7 | 77,2% | 0,919 (turun 1%) |
| 0,5 | 32,8% | 0,758 (turun 18%) |
| 0,3 | 5,6% | 0,544 (turun 41%) |

`alpha ≈ 0,7` adalah titik paling seimbang: 23% hasil sudah bergeser ke preferensi pelanggan, tapi relevansi query nyaris tidak turun (~1%). Di bawah `alpha = 0,5`, relevansi mulai anjlok.

Catatan penting: fitur ini baru bisa dipakai untuk 2.876 dari 99.441 pelanggan (2,9%) yang punya histori ≥2 order. Sebagian besar pelanggan Olist cuma belanja sekali, jadi >97% pelanggan belum bisa menikmati personalisasi ini.

## 8. Rekomendasi

Tiap rekomendasi di bawah menjawab salah satu business question di Bagian 1.

**Untuk BQ 1 & 2  kualitas pencarian:**

1. Gabungkan TF-IDF dan semantic jadi satu (hybrid), jangan pilih salah satu. Semantic menambal miss TF-IDF, TF-IDF menambal salah arah semantic saat kosakata query jauh dari profil merchant.
2. Tulis profil merchant dengan kalimat yang lebih natural, bukan template kaku, supaya model semantic punya lebih banyak "bahan" untuk memahami makna  ini langsung menaikkan recall semantic (BQ2) dan mengurangi salah arah TF-IDF (BQ1).

**Untuk BQ 3  personalisasi:**

3. Pakai `alpha` di kisaran 0,6–0,8, bukan asal pasang 0,7 tanpa dites  rentang ini menjaga relevansi query tetap turun di bawah ~5%, berdasarkan hasil pada query uji di atas.
4. Sebelum rollout, ulangi pengujian trade-off `alpha` di beberapa query lain (bukan cuma "produk elektronik") untuk memastikan titik seimbang 0,6–0,8 juga berlaku di sana.
5. Buat cara personalisasi lain berbasis aktivitas sesi (kategori yang sedang dilihat/diklik), khusus untuk >97% pelanggan yang belum punya histori order dan belum kejangkau blending `alpha`.

## 9. Batasan & Keterbatasan Proyek

1. Ground truth evaluasi (Recall@K/Precision@K/MRR) masih dibuat sintetis dari taksonomi kategori, bukan dari log pencarian pengguna asli.
2. Profil merchant hasil rekonstruksi dari data transaksi, bukan deskripsi asli yang ditulis merchant.
3. ~31% seller (957 dari 3.095) tidak masuk pencarian karena order-nya kurang dari 3  datanya terlalu sedikit untuk bikin profil yang representatif.
4. Evaluasi trade-off personalisasi di bagian 7.3, meski terukur lintas 50 pelanggan × 4 nilai alpha, baru dicoba pada **1 query** ("produk elektronik") dan **1 engine** (semantic)  belum divariasikan ke query atau engine lain.

## 10. Cara Menjalankan

1. Taruh 9 file CSV Olist di folder `data/` (satu level di atas atau sejajar dengan folder `notebooks/`).
2. Install dependency:
   ```bash
   pip install pandas numpy scikit-learn sentence-transformers faiss-cpu matplotlib
   ```
3. Jalankan notebook `semantic_search_merchant.ipynb` dari atas ke bawah (Run All). Kalau tidak ada akses internet ke `huggingface.co`, engine semantic otomatis pindah ke TF-IDF+SVD (LSA)  tetap jalan, tapi kualitasnya lebih rendah dari model transformer asli.
4. (Opsional) jalankan demo interaktif terpisah:
   ```bash
   streamlit run app.py
   ```

## 11. Struktur Proyek

```
.
├── data/                              # 9 CSV Olist (tidak disertakan di repo)
├── notebooks/
│   └── semantic_search_merchant.ipynb # notebook utama (pipeline + evaluasi + insight)
├── assets/                            # chart hasil evaluasi untuk README
├── app.py                             # demo Streamlit (terpisah dari notebook)
└── README.md
```

## 12. Pengembangan Lanjutan

1. Kumpulkan log pencarian pengguna asli lewat demo Streamlit, supaya ground truth evaluasi lebih mencerminkan kondisi nyata.
2. Coba model embedding multibahasa lain, atau fine-tuning ringan khusus e-commerce Indonesia.
3. A/B test hybrid scoring (TF-IDF + semantic) melawan semantic murni, pakai query nyata.
4. Perluas evaluasi personalisasi ke lebih banyak query dan nilai `alpha` yang lebih granular, untuk menemukan titik optimal per kategori.