# Laporan Proyek Machine Learning - Deden Sholeh Hermawan

## Domain Proyek
Dalam era ledakan informasi digital, pengguna sering kali kesulitan memilih konten yang relevan di antara ribuan pilihan yang tersedia. Di industri hiburan, khususnya perfilman, sistem rekomendasi menjadi krusial untuk meningkatkan pengalaman pengguna (user experience) dan mempertahankan retensi pelanggan. Proyek ini berfokus pada pembangunan sistem rekomendasi film menggunakan pendekatan Collaborative Filtering berbasis Deep Learning.
Sistem rekomendasi berbasis Collaborative Filtering menjadi solusi standar industri karena efektivitasnya dalam memanfaatkan "kebijaksanaan orang banyak" (wisdom of the crowd). Teknik ini mengasumsikan bahwa jika pengguna A memiliki minat yang sama dengan pengguna B dalam beberapa hal, maka mereka cenderung memiliki minat yang sama pada hal lainnya. Penelitian menunjukkan bahwa pendekatan Deep Learning dalam Collaborative Filtering mampu menangkap hubungan non-linear yang lebih kompleks antara pengguna dan item dibandingkan metode matriks konvensional.
Referensi: [1] F. Ricci, L. Rokach, and B. Shapira, Introduction to Recommender Systems Handbook. Boston, MA: Springer US, 2011. [2] J. Bobadilla, F. Ortega, A. Hernando, and A. Gutiérrez, “Recommender systems survey,” Knowledge-Based Systems, vol. 46, pp. 109–132, Jul. 2013.

## Business Understanding
Tujuan utama proyek ini adalah menyediakan sistem yang mampu memberikan rekomendasi film yang dipersonalisasi.
Bagian laporan ini mencakup:

### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara mengolah data interaksi pengguna dan film yang bersifat kategorikal agar dapat diproses secara efisien oleh model Deep Learning?
- Bagaimana membangun model Collaborative Filtering yang dapat memberikan rekomendasi yang personal dengan tingkat akurasi (error) yang rendah?
- Bagaimana cara mengatasi risiko overfitting agar model tetap memiliki kemampuan prediksi yang baik pada data yang belum pernah dilihat sebelumnya?
  
### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Transformasi Fitur (Encoding): Menyandikan userId dan movieId ke dalam indeks numerik agar dapat diproses oleh layer embedding pada model deep learning. Tanpa tahap ini, model tidak bisa membaca data ID yang bersifat kategorikal.
- Prediksi Rating Akurat: Membangun model yang mampu memprediksi skor rating film dengan tingkat error (RMSE) seminimal mungkin. Hal ini bertujuan agar hasil prediksi mendekati penilaian asli pengguna di dunia nyata.
- Top-10 Recommendation: Menghasilkan daftar 10 rekomendasi film yang belum pernah ditonton pengguna. Rekomendasi ini didasarkan pada skor prediksi tertinggi yang dihasilkan oleh model untuk meningkatkan relevansi saran film.
Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

    ### Solution statements
    A.  Pembangunan Model Collaborative Filtering (RecommenderNet).Membangun arsitektur jaringan saraf (Neural Network) yang memanfaatkan lapisan Embedding untuk merepresentasikan hubungan antara user dan item (film).Teknik ini bekerja dengan cara:
      1. Mengonversi userId dan movieId ke dalam vektor berdimensi rendah (50 dimensi) agar model dapat mempelajari fitur laten (preferensi tersembunyi).
      2. Menggunakan operasi Dot Product antara vektor user dan item untuk menghitung skor kecocokan.
      3. Menggunakan fungsi aktivasi Sigmoid pada output akhir untuk menghasilkan prediksi skor dalam rentang 0 hingga 1 (selaras dengan data rating yang telah dinormalisasi).    
    B.  Optimalisasi Model dengan Hyperparameter Tuning & Regularization Melakukan peningkatan (improvement) pada model dasar agar hasil prediksi lebih akurat dan tidak overfitting melalui:
      1. Optimizer Adam: Menggunakan pengoptimal Adam dengan learning rate sebesar 0.001 untuk mempercepat konvergensi selama pelatihan.
      2. Regularisasi L2: Menambahkan parameter embeddings_regularizer (L2) pada lapisan embedding untuk mencegah model terlalu menghafal data latihan (overfitting).
      3. Evaluasi RMSE: Mengukur performa model menggunakan metrik Root Mean Squared Error (RMSE). Target solusinya adalah mencapai nilai RMSE yang terus menurun secara stabil baik pada data training maupun validation.
    
## Data Understanding
Dataset yang digunakan dalam proyek ini adalah MovieLens 100K Dataset (ml-latest-small). Dataset ini merupakan kumpulan data rating film yang dikumpulkan oleh GroupLens Research. Data ini umum digunakan dalam riset sistem rekomendasi untuk memodelkan interaksi antara pengguna dan item (film).
Tautan Unduh: Dataset yang digunakan berasal dari [Kaggle - MovieLens Small](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)
Jumlah Data: Memiliki 100.836 rating dari 610 pengguna terhadap 9.742 film.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- userId: Identitas unik untuk setiap pengguna (integer).
- movieId: Identitas unik untuk setiap film yang diberikan rating (integer).
- rating: Skor yang diberikan pengguna dalam skala 0.5 hingga 5.0 (float).
- timestamp: Waktu saat rating diberikan dalam format unix seconds.
- movieId: Identitas unik untuk setiap film (sesuai dengan yang ada di ratings.csv).
- title: Judul film beserta tahun rilisnya.
- genres: Kategori atau genre film (seperti Action, Comedy, Drama, dll).    

**Rubrik/Kriteria Tambahan (Opsional)**:
- Analisis Sebaran Rating: Dilakukan visualisasi histogram untuk melihat frekuensi nilai rating. Hasilnya menunjukkan bahwa nilai 4.0 merupakan rating yang paling sering diberikan oleh pengguna.
- Identifikasi Nilai Unik: Menghitung jumlah entitas unik untuk menentukan ukuran matriks embedding. Didapatkan angka 610 pengguna unik dan 9.724 film unik.
- Kualitas Data: Memeriksa keberadaan data kosong menggunakan info(). Hasil menunjukkan dataset dalam kondisi bersih tanpa missing values, sehingga tidak diperlukan imputasi data.

## Data Preparation
Tahap ini bertujuan untuk mentransformasi data mentah menjadi format yang siap digunakan oleh model Neural Network. Berikut adalah langkah-langkah yang dilakukan:
1. Feature Encoding
   - Proses: Mengonversi ID unik ke dalam indeks integer (0 hingga jumlah total user/movie).
   - Alasan: Model Deep Learning memerlukan input berupa indeks integer berurutan agar dapat dipetakan ke dalam Layer Embedding. Tanpa proses ini, model tidak bisa memproses ID yang nilainya meloncat atau tidak beraturan.
2. Normalisasi Rating
   - Proses: Menggunakan rumus Min-Max Scaling: $y = \frac{x - \min}{\max - \min}$.
   - Alasan: Normalisasi mempercepat proses konvergensi saat pelatihan model (training). Skala 0-1 sangat ideal untuk fungsi aktivasi sigmoid yang digunakan di lapisan output model.
3. Data Shufflin
   - Alasan: Untuk memastikan bahwa urutan data tidak mempengaruhi proses belajar model, sehingga model dapat mempelajari pola secara objektif dari seluruh distribusi data.
4. Splitting Data (Pembagian Data)
   - 80% Data Training: Digunakan agar model mempelajari pola hubungan antara user dan film.
   - 20% Data Validation: Digunakan untuk menguji performa model pada data yang belum pernah dilihat sebelumnya guna mendeteksi overfitting.

## Modeling
Tahapan ini menggunakan algoritma Collaborative Filtering dengan arsitektur Neural Network.
RecommenderNet
Model yang dibangun menggunakan class RecommenderNet dengan library Keras/TensorFlow.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi.
- Cara Kerja: Model menghitung skor kecocokan antara pengguna dan film menggunakan teknik Embedding. Embedding merepresentasikan setiap user dan film sebagai vektor angka dalam ruang multidimensi (50 dimensi).
- Parameter:
a. embedding_size = 50: Jumlah dimensi laten.
b. Adam Optimizer: Digunakan sebagai pengoptimal dengan learning rate 0.001.
c. Binary Crossentropy: Digunakan sebagai fungsi kerugian (loss function) karena rating telah dinormalisasi ke rentang 0-1.
Kelebihan: Dapat menangkap pola preferensi yang sangat kompleks dan tersembunyi (latent features) antara pengguna dan film. Kekurangan: Memerlukan komputasi yang lebih intensif dibandingkan algoritma statistik sederhana dan membutuhkan data interaksi yang cukup banyak.
Kekurangan: Model menghadapi kendala Cold Start, yaitu kesulitan memberikan rekomendasi akurat bagi pengguna atau film baru yang belum memiliki riwayat rating. Selain itu, proses pelatihannya membutuhkan sumber daya komputasi yang lebih besar dan waktu yang lebih lama.

## Evaluation
Pada tahap evaluasi, proyek ini menggunakan metrik Root Mean Squared Error (RMSE). Metrik ini dipilih karena relevan untuk mengukur kinerja model regresi atau prediksi angka (rating) dalam sistem rekomendasi Collaborative Filtering.
Penjelasan Metrik RMSE
Root Mean Squared Error (RMSE) adalah metrik yang digunakan untuk mengukur seberapa besar penyimpangan atau rata-rata kesalahan hasil prediksi model terhadap nilai aktual (nilai asli).
1. Cara Kerja Metrik:
   - RMSE menghitung selisih antara rating yang diprediksi oleh model dengan rating asli yang diberikan oleh pengguna.
   - Selisih tersebut dikuadratkan (agar nilai negatif tidak saling menghilangkan), lalu dirata-ratakan, dan terakhir diakarkan untuk mengembalikan nilai ke satuan yang sama dengan rating asli.
   - Semakin kecil nilai RMSE, semakin akurat model tersebut. Dalam konteks proyek ini, RMSE yang rendah berarti prediksi rating model sangat mendekati preferensi nyata pengguna.
2. Hasil Proyek Berdasarkan Metrik
   - Stabilitas Pelatihan: Grafik evaluasi menunjukkan bahwa nilai RMSE pada data pelatihan (train) dan data validasi (val) mengalami penurunan secara konsisten. Hal ini mengindikasikan bahwa model belajar dengan baik dari pola data.
   - Kualitas Prediksi: Model mencapai nilai RMSE akhir yang cukup rendah (misalnya: di bawah 0.20 pada skala rating yang dinormalisasi). Nilai ini menunjukkan bahwa perbedaan antara rating prediksi dan rating aktual sangat tipis.
   - Visualisasi Hasil: Jarak yang kecil antara kurva training dan validation pada akhir epoch menunjukkan bahwa model memiliki kemampuan generalisasi yang baik dan tidak mengalami overfitting yang parah.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi


**---Ini adalah bagian akhir laporan---**

1. Berdasarkan proses pelatihan, berikut adalah grafik konvergensi metrik RMSE pada data training dan validation:
Dari grafik di bawah, terlihat bahwa model mengalami penurunan error yang stabil, yang menunjukkan proses pembelajaran berjalan dengan baik.
<img width="727" height="577" alt="Screenshot 2026-01-31 123935" src="https://github.com/user-attachments/assets/49d05695-c670-4b1f-8020-32615b81c94c" />

2. Berikut adalah hasil uji coba model dalam memberikan rekomendasi kepada pengguna:
<img width="727" height="577" alt="<img width="963" height="485" alt="Screenshot 2026-01-31 124822" src="https://github.com/user-attachments/assets/89e8a6a6-415b-4644-a674-0a85c5f2141e" />
" src="https://github.com/user-attachments/assets/49d05695-c670-4b1f-8020-32615b81c94c" />
