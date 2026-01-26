# Laporan Proyek Machine Learning - Deden Sholeh Hermawan

## Project Overview
Perkembangan layanan musik digital seperti Spotify, Apple Music, Joox, dan YouTube Music telah meningkatkan akses pengguna terhadap jutaan lagu dalam berbagai genre dan bahasa. Meskipun memberikan kemudahan, jumlah opsi yang sangat besar dapat menyebabkan information overload, yaitu kondisi ketika pengguna kesulitan memilih lagu yang sesuai preferensinya. Oleh karena itu diperlukan sistem rekomendasi untuk membantu pengguna menemukan musik yang relevan secara lebih efisien.

Sistem rekomendasi banyak digunakan pada platform komersial modern dan telah terbukti meningkatkan user engagement, kepuasan pengguna, serta waktu penggunaan platform (Ricci et al., 2011). Pada proyek ini dibangun sistem rekomendasi musik menggunakan dua pendekatan algoritmik yang berbeda, yaitu Content-Based Filtering dan Collaborative Filtering menggunakan Neural Collaborative Filtering (NCF) dengan basis dataset Spotify.

**Rubrik/Kriteria Tambahan (Opsional)**:
Pengguna layanan musik seperti Spotify sering mengalami kesulitan memilih lagu karena banyaknya pilihan (information overload). Sistem rekomendasi digunakan untuk membantu memberikan saran lagu yang sesuai preferensi pengguna. Pada proyek ini diterapkan dua pendekatan, yaitu Content-Based Filtering dan Collaborative Filtering, menggunakan dataset Spotify dari Kaggle.(Ricci et al., 2011)

## Business Understanding
Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.
Bagian laporan ini mencakup:

### Problem Statements
Proyek ini menjawab pertanyaan berikut:
1. Bagaimana memberikan rekomendasi lagu yang mirip berdasarkan karakteristik konten seperti artis dan genre?
2. Bagaimana memberikan rekomendasi lagu berdasarkan pola preferensi pengguna lain tanpa menggunakan fitur konten?
3. Metode mana yang menghasilkan rekomendasi yang lebih relevan, beragam, dan cocok untuk konteks tertentu?
   
### Goals
Tujuan proyek ini adalah:
1.Mengimplementasikan Content-Based Filtering untuk memberikan rekomendasi lagu mirip berdasarkan artis dan genre.
2.Mengimplementasikan Collaborative Filtering untuk memberikan rekomendasi musik berdasarkan implicit feedback.
3.Membandingkan kedua pendekatan berdasarkan hasil rekomendasi dan evaluasi metrik.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
   Dua solusi yang diajukan adalah:
1. Content-Based Filtering
menggunakan TF-IDF + Cosine Similarity
merekomendasikan lagu yang mirip secara konten
2. Collaborative Filtering (NCF)
menggunakan embedding user–item dan multilayer perceptron
merekomendasikan lagu berdasarkan pola kesukaan pengguna lain

## Data Understanding
Dataset yang digunakan adalah Spotify Tracks Dataset yang tersedia di Kaggle pada tautan. Sumber atau tautan untuk mengunduh dataset, Link 
[GitHub Pages]([https://pages.github.com/](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)).
Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- track_id : ID unik track Spotify
- artists : nama artis atau kolaborasi artis
- track_name : judul lagu
- track_genre : genre musik
- popularity : skor popularitas berbasis Spotify
- danceability : tingkat kelayakan untuk menari
- energy : intensitas energi lagu
- valence : mood positif
- tempo : BPM (beats per minute)
- speechiness : proporsi suara percakapan
- instrumentalness : proporsi instrumental
- duration_ms : durasi dalam milidetik

**Rubrik/Kriteria Tambahan (Opsional)**:
Dataset Spotify memiliki lebih dari 100.000 lagu dengan 21 fitur yang berisi informasi terkait nama lagu, artis, genre, popularitas, dan karakteristik audio. Dataset relatif bersih dengan jumlah missing value yang sangat kecil. Popularitas lagu cenderung tidak merata dan menunjukkan pola long-tail, yaitu hanya sebagian kecil lagu yang memiliki popularitas tinggi. Variasi genre juga cukup luas sehingga mendukung penggunaan sistem rekomendasi. Secara keseluruhan, dataset sudah cukup representatif dan siap digunakan pada tahap pemodelan.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. 
Tahapan data preparation meliputi:
1. Drop kolom tidak relevan
- Unnamed: 0 Kolom ini tidak mengandung informasi dan tidak diperlukan untuk proses analisis sehingga dihapus.
2. Handling Missing Value
- missing sangat kecil → dilakukan dropna()
3. Normalisasi fitur numerik
- danceability, energy, valence, tempo → MinMaxScaler
- alasan: untuk cosine similarity & neural network
4. Implicit Rating Creation
- popularity → rating (1–5) menggunakan min-max normalization
- karena tidak tersedia rating eksplisit
5. Encoding
- artists → user_id
- track_id → item_id
- diperlukan untuk embedding pada model CF
  
**Rubrik/Kriteria Tambahan (Opsional)**: 
Tahap data preparation dilakukan untuk menyiapkan dataset agar dapat diproses oleh model. Langkah-langkah yang dilakukan meliputi penghapusan kolom yang tidak diperlukan, penanganan missing value, dan konversi popularitas menjadi implicit rating agar dapat digunakan pada model collaborative filtering. Selain itu dilakukan encoding terhadap artists dan track_id menjadi user_id dan item_id untuk mendukung proses embedding. Tahapan ini diperlukan agar data memiliki format yang sesuai dan konsisten dengan kebutuhan algoritma sistem rekomendasi.

## Modeling
Pada proyek ini digunakan dua pendekatan sistem rekomendasi yang berbeda, yaitu Content-Based Filtering dan Collaborative Filtering. Kedua pendekatan tersebut diterapkan pada dataset Spotify untuk menghasilkan rekomendasi musik dalam bentuk Top-N Recommendation.

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. Content-Based Filtering 
Content-Based Filtering memberikan rekomendasi berdasarkan kemiripan antar lagu. Kemiripan dihitung menggunakan fitur yang berasal dari konten musik, dalam hal ini:
- artists
- track_genre
Tahapan yang dilakukan meliputi:
a. Menggabungkan fitur artists + track_genre menjadi satu fitur teks (content_feature)
b. Melakukan TF-IDF vectorization
c. Menghitung cosine similarity
d. Memilih lagu acuan (anchor track) berdasarkan artis yang dipilih secara acak
e. Mengambil Top-10 track dengan similarity tertinggi
Output Top-N Recommendation (CB)
- Output berupa daftar 10 lagu yang paling mirip dengan lagu acuan. Rekomendasi ini bersifat homogen, sering berasal dari artis/genre yang sama atau berdekatan.
  
2. Collaborative Filtering (Neural Collaborative Filtering)
Collaborative Filtering memberikan rekomendasi berdasarkan pola preferensi pengguna lain. Karena dataset tidak memiliki rating eksplisit, digunakan implicit rating yang dikonversi dari fitur popularity ke skala 1–5 menggunakan min-max normalization.
Model yang digunakan adalah Neural Collaborative Filtering (NCF) yang mengkombinasikan:
- User Embedding
- Item Embedding
- Multilayer Perceptron (MLP)
Arsitektur model terdiri dari:
- Embedding layer untuk user (dim = 32)
- Embedding layer untuk item (dim = 32)
- Dense layer: 64 → 32 → 1
- Optimizer: Adam
- Loss function: MSE
- Evaluation metric: RMSE
Pendekatan ini bersifat non-linear sehingga mampu belajar representasi laten interaksi user–item lebih baik dibanding metode kollaboratif klasik.
Proses
Langkah model CF:
a. User diwakili oleh user_id (hasil encoding artis)
b. Item diwakili oleh item_id (hasil encoding track)
c. Dataset dibentuk dalam format (user_id, item_id, rating)
d. Data dibagi menjadi train/test (80/20)
e. Model dilatih pada implicit feedback
f. Prediksi dilakukan pada seluruh item untuk user target
g. Diambil Top-10 item dengan prediksi rating tertinggi
Output Top-N Recommendation (CF)
- Output berupa daftar 10 lagu dengan skor prediksi rating tertinggi untuk user tertentu. Rekomendasi ini bersifat lebih variatif dibanding content-based karena tidak bergantung pada kesamaan konten, tetapi pada pola preferensi kolektif.

Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.
Menurut Ricci et al. (2011), Content-Based Filtering unggul pada skenario cold-start, sedangkan Collaborative Filtering unggul dalam diversifikasi rekomendasi melalui item discovery berbasis preferensi kolektif pengguna.
1. Content-Based → cocok untuk cold-start, interpretasi mudah, tetapi rekomendasi cenderung homogen.
2. Collaborative → unggul pada exploration & diversity, namun bergantung pada data interaksi dan lemah pada cold-start.
   
## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
Pada proyek ini digunakan metrik Root Mean Square Error (RMSE) untuk mengevaluasi model Collaborative Filtering karena model memprediksi nilai rating untuk setiap pasangan user-item. RMSE menghitung rata-rata galat kuadrat antara nilai aktual dan prediksi, kemudian diambil akar kuadratnya agar kembali dalam skala asli rating. Karena terdapat operasi kuadrat, RMSE memberikan penalti lebih besar terhadap error yang besar, sehingga bersifat sensitif terhadap deviasi prediksi yang jauh dari nilai sebenarnya.

**---Ini adalah bagian akhir laporan---**
References :
Ricci, F., Rokach, L., & Shapira, B. (2011). Recommender Systems Handbook. Springer.
Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems, 6(4), 1–19.
Su, X., & Khoshgoftaar, T. M. (2009). A Survey of Collaborative Filtering Techniques. Advances in Artificial Intelligence, 2009, 1–19.
Bawden, D., & Robinson, L. (2009). The Dark Side of Information: Overload, Anxiety and Other Paradoxes and Pathologies. Journal of Information Science, 35(2), 180–191.
Pandya, M. (2020). Spotify Tracks Dataset. Kaggle.
