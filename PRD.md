# PRD — Library Python Aspect-Based Sentiment Analysis (ABSA) Teks Bahasa Indonesia Informal Berbasis Multi-Engine

| Atribut | Keterangan |
|---|---|
| **Nama Produk (kerja)** | `absa-library` (nama final PyPI perlu dicek ketersediaannya) |
| **Nama Paket Import** | `absa_library` |
| **Versi Dokumen** | 1.0 (Draft) |
| **Status** | Draft untuk review dosen pembimbing |
| **Penulis** | Yusuf Rahmadhani Asy'Ari (NRP 3126640042) |
| **Pembimbing** | Prof. Ali Ridho Barakbah, S.Kom., Ph.D.; Entin Martiana Kusumaningtyas, S.Kom., M.Kom |
| **Institusi** | Politeknik Elektronika Negeri Surabaya (PENS) — D4 Teknik Informatika |
| **Sumber** | Proposal Proyek Akhir 2026 |

> **Catatan membaca dokumen.** Bagian yang **berasal langsung dari proposal** ditulis apa adanya. Bagian yang merupakan **asumsi / keputusan desain tambahan** dari saya ditandai dengan label **[ASUMSI]** dan dikumpulkan kembali di Bagian 18 (Open Questions) agar bisa dikonfirmasi dengan pembimbing.

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Latar Belakang dan Problem Statement](#2-latar-belakang-dan-problem-statement)
3. [Tujuan, Non-Tujuan, dan Metrik Keberhasilan](#3-tujuan-non-tujuan-dan-metrik-keberhasilan)
4. [Target Pengguna dan Persona](#4-target-pengguna-dan-persona)
5. [User Stories](#5-user-stories)
6. [Ruang Lingkup](#6-ruang-lingkup)
7. [Gambaran Solusi dan Arsitektur](#7-gambaran-solusi-dan-arsitektur)
8. [Functional Requirements](#8-functional-requirements)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Spesifikasi API Publik](#10-spesifikasi-api-publik)
11. [Desain Kelas dan Struktur Proyek](#11-desain-kelas-dan-struktur-proyek)
12. [Spesifikasi Data dan Dataset](#12-spesifikasi-data-dan-dataset)
13. [Strategi Pengujian dan Evaluasi](#13-strategi-pengujian-dan-evaluasi)
14. [Packaging, Distribusi, dan Dokumentasi](#14-packaging-distribusi-dan-dokumentasi)
15. [Dependensi dan Lingkungan](#15-dependensi-dan-lingkungan)
16. [Milestone dan Rencana Pengerjaan](#16-milestone-dan-rencana-pengerjaan)
17. [Risiko dan Mitigasi](#17-risiko-dan-mitigasi)
18. [Open Questions dan Asumsi](#18-open-questions-dan-asumsi)
19. [Definition of Done dan Acceptance Criteria](#19-definition-of-done-dan-acceptance-criteria)
20. [Lampiran](#20-lampiran)

---

## 1. Ringkasan Eksekutif

Proyek ini mengembangkan **library Python open-source yang dapat diinstal via `pip`** untuk **Aspect-Based Sentiment Analysis (ABSA) pada teks ulasan berbahasa Indonesia informal**. Library menyediakan **empat mesin klasifikasi** yang dapat dipilih secara dinamis (runtime): **Rule-Based, Naive Bayes, SVM, dan IndoBERT**, dilengkapi **modul auto-preprocessing** untuk teks informal (slang, singkatan, typo, perulangan karakter).

Pembeda utama proyek ini **bukan** pada penemuan algoritma baru, melainkan pada **rekayasa perangkat lunak untuk AI (Software Engineering for AI)**: mengubah skrip Jupyter Notebook yang monolitik menjadi **paket modular, terdokumentasi, teruji (coverage > 80%), dan siap pakai** dengan API setingkat scikit-learn.

**Arsitektur inti:** Facade Pattern (`AspectAnalyzer`) + Strategy Pattern (`BaseEngine` dan empat turunannya), dengan Lazy Loading untuk bobot IndoBERT.

**Contoh penggunaan target:**

```python
from absa_library import AspectAnalyzer

analyzer = AspectAnalyzer(engine="svm")
hasil = analyzer.analyze(["Barangnya bagus banget, tapi kurirnya lama"])
print(hasil)
```

---

## 2. Latar Belakang dan Problem Statement

### 2.1 Latar belakang

- Volume ulasan konsumen di platform digital/e-commerce Indonesia sangat besar dan menjadi dasar *data-driven decision making*.
- ABSA diperlukan karena satu ulasan sering memuat sentimen campuran (contoh: *"Makanannya enak, tetapi pelayanannya lambat sekali"* → Makanan: positif, Pelayanan: negatif).
- Teks Indonesia informal sulit diolah: singkatan tidak baku, slang, typo, karakter berulang, dan sentimen campuran.

### 2.2 Problem statement

| # | Masalah | Dampak |
|---|---|---|
| P1 | **Software engineering gap**: mayoritas riset/eksperimen ABSA berupa skrip Jupyter Notebook yang monolitik, sekuensial, kaku, dan tidak modular. | Sulit digunakan ulang dan diintegrasikan ke aplikasi nyata. |
| P2 | Pengguna harus **menulis ulang preprocessing**, mengelola dependensi, dan melatih ulang model dari nol setiap kali bereksperimen. | Waktu terkuras untuk *boilerplate code*, bukan analisis. |
| P3 | **Tidak ada library** khusus ABSA Bahasa Indonesia yang terstruktur, terpaket, dan siap pakai (setara scikit-learn/pandas dalam hal kemudahan API). | Hambatan adopsi bagi mahasiswa, peneliti, dan praktisi. |
| P4 | Penelitian terkait (Mustakim & Priyanta 2022; Lestarie 2024; Hammami dkk. 2026; Dadap dkk. 2023; Renaldi 2025) berhenti pada skrip eksperimental dan/atau spesifik satu domain. | Hasil tidak reusable lintas domain. |

### 2.3 Peluang

Library yang **general (lintas domain)**, **modular**, dan **pip-installable** mengisi celah yang ditinggalkan penelitian sebelumnya, sehingga peneliti lain dapat langsung memakainya untuk data sendiri.

---

## 3. Tujuan, Non-Tujuan, dan Metrik Keberhasilan

### 3.1 Tujuan produk

| ID | Tujuan |
|---|---|
| G1 | Menyediakan library Python yang dapat diinstal dengan `pip install` dan digunakan dengan beberapa baris kode. |
| G2 | Menyediakan **4 engine** (Rule-Based, Naive Bayes, SVM, IndoBERT) yang dapat **diganti saat runtime** tanpa mengubah kode utama. |
| G3 | Menyediakan **auto-preprocessing** untuk teks informal Bahasa Indonesia, termasuk spell correction (Damerau-Levenshtein) dan normalisasi slang. |
| G4 | Menjamin kualitas melalui **unit test dengan code coverage > 80%**. |
| G5 | Menyediakan fungsi **evaluasi** (akurasi, presisi, recall, F1, waktu eksekusi) untuk membandingkan keempat engine. |
| G6 | Menerapkan **OOP, Facade Pattern, dan Strategy Pattern** dengan arsitektur modular. |
| G7 | Menyediakan penanganan error yang aman (input validation, safe-state recovery, logging). |

### 3.2 Non-tujuan (out of scope)

- Menemukan algoritma/rumus AI baru.
- Membangun aplikasi web/GUI/dashboard atau REST API server.
- Mendukung bahasa selain Bahasa Indonesia.
- Mendukung ekstraksi aspek *implicit* tingkat lanjut (aspek tidak disebut eksplisit) **[ASUMSI]**.
- Pelatihan model dalam skala besar/distributed.

### 3.3 Metrik keberhasilan

| ID | Metrik | Target | Cara Ukur |
|---|---|---|---|
| M1 | Code coverage | **> 80%** | `pytest --cov` |
| M2 | Unit test lulus | 100% test pass, tanpa error fatal | CI / `pytest` |
| M3 | Instalasi | `pip install` berhasil dari PyPI (atau TestPyPI) di Python bersih | Uji manual + CI |
| M4 | Jumlah baris kode untuk use case dasar | ≤ 5 baris (import → hasil) | Demonstrasi Notebook |
| M5 | Robustness input | Input kosong/angka/teks ekstrem **tidak menyebabkan crash**, mengembalikan pesan error informatif | Test validasi input |
| M6 | Recovery | Kegagalan unduh IndoBERT tercatat di Logger dan sistem kembali ke safe state | Test simulasi gagal unduh |
| M7 | Perbandingan engine | Akurasi, presisi, recall, F1, dan waktu eksekusi seluruh engine tersedia di Bab 4 | Evaluator pada data uji 20% |
| M8 | Lazy loading | Bobot IndoBERT **tidak** diunduh/dimuat saat engine lain dipakai atau saat `import` | Test + pengukuran waktu import |

> Proposal **tidak** menetapkan target akurasi minimum untuk model. Target akurasi dibiarkan sebagai hasil eksperimen (lihat Open Question OQ-9).

---

## 4. Target Pengguna dan Persona

| Persona | Latar belakang | Kebutuhan utama | Fitur kunci |
|---|---|---|---|
| **Mahasiswa data mining/NLP** | Belajar algoritma klasifikasi; kurang nyaman dengan preprocessing & boilerplate | Fokus memahami & membandingkan algoritma | `analyze()`, `evaluate()`, pergantian engine mudah |
| **Dosen/pengajar** | Menyiapkan praktikum ABSA | Alat ajar yang konsisten dan reproducible | API sederhana, dokumentasi, contoh Notebook |
| **Peneliti** | Menguji algoritma ABSA pada dataset sendiri | Membandingkan engine pada data sendiri tanpa menulis ulang pipeline | Multi-engine, Evaluator, `fit()` pada data sendiri |
| **Praktisi data / developer** | Mengintegrasikan ABSA ke aplikasi/pipeline | API stabil, error handling, logging, packaging | Facade API, return `dict`/`DataFrame`, validasi input |

Lingkungan kerja utama: **Jupyter Notebook / Google Colab** (sesuai *User Workflow* proposal).

---

## 5. User Stories

| ID | Sebagai… | Saya ingin… | Supaya… | Prioritas |
|---|---|---|---|---|
| US-01 | Pengguna | menginstal library dengan satu perintah `pip install` | tidak perlu mengatur proyek manual | Must |
| US-02 | Pengguna | membuat objek `AspectAnalyzer(engine="svm")` dan memanggil `analyze(texts)` | mendapat aspek, sentimen, dan confidence tanpa menulis preprocessing | Must |
| US-03 | Pengguna | memasukkan satu string **atau** list string | fleksibel untuk satu kalimat maupun batch | Must |
| US-04 | Pengguna | mengganti engine (rule-based / naive-bayes / svm / indobert) dengan mengubah satu parameter | membandingkan algoritma dengan cepat | Must |
| US-05 | Pengguna | teks slang/typo/singkatan dibersihkan otomatis | akurasi tidak turun karena noise | Must |
| US-06 | Peneliti | memanggil `evaluate(x_test, y_test)` | mendapat akurasi, presisi, recall, F1 | Must |
| US-07 | Peneliti | melihat waktu eksekusi tiap engine | membandingkan efisiensi | Must |
| US-08 | Pengguna | hasil dalam bentuk `list[dict]` atau `DataFrame` | mudah dianalisis lanjut dengan pandas | Must |
| US-09 | Developer | library tidak crash saat input kosong/salah tipe | aman diintegrasikan | Must |
| US-10 | Pengguna | model IndoBERT hanya diunduh saat pertama kali dibutuhkan | hemat memori, waktu import, dan bandwidth | Must |
| US-11 | Developer | kegagalan unduh/proses tercatat di log tanpa menghentikan runtime | mudah debugging & tetap stabil | Must |
| US-12 | Peneliti | melatih (`fit`) Naive Bayes/SVM/IndoBERT pada dataset sendiri | menyesuaikan dengan domain saya | Should **[ASUMSI]** |
| US-13 | Peneliti | menyimpan & memuat model terlatih (`save`/`load`) | tidak melatih ulang tiap sesi | Should **[ASUMSI]** |
| US-14 | Pengguna | mendefinisikan daftar aspek/kamus aspek sendiri | library cocok untuk domain berbeda | Should **[ASUMSI]** |
| US-15 | Pengguna | menambah kamus slang kustom | menangani istilah domain/komunitas | Could **[ASUMSI]** |

---

## 6. Ruang Lingkup

### 6.1 In scope

1. Modul **TextPreprocessor**: cleaning, case folding, tokenisasi, spell correction (Damerau-Levenshtein), normalisasi slang/singkatan, penghapusan stop words.
2. Kelas abstrak **BaseEngine** + 4 engine: `RuleBasedEngine`, `NaiveBayesEngine`, `SVMEngine`, `IndoBERTEngine`.
3. **ModelManager** untuk Lazy Loading model dari Hugging Face Hub.
4. Facade **AspectAnalyzer** dengan `analyze()` dan `evaluate()`.
5. **Evaluator** (akurasi, presisi, recall, F1, confusion matrix, waktu eksekusi, simpan laporan).
6. **Logger** untuk error handling dan pencatatan proses.
7. Validasi input dan error handling (try-except, safe state).
8. Unit test (pytest), coverage > 80%.
9. Packaging (`pyproject.toml`, `python -m build` → `.whl` + `.tar.gz`) dan distribusi ke PyPI.
10. Dokumentasi (folder `docs/`) dan Notebook contoh.
11. Eksperimen pada dataset riil (publik + scraping) untuk Bab 4.

### 6.2 Out of scope

- UI/dashboard, REST API, deployment cloud.
- Scraper sebagai bagian dari library (scraping hanya dilakukan **di luar library** untuk pengumpulan dataset penelitian) **[ASUMSI]**.
- Bahasa selain Indonesia.

---

## 7. Gambaran Solusi dan Arsitektur

### 7.1 Arsitektur tiga bagian (sesuai Gambar 3.1 proposal)

```
┌──────────────────────── BAGIAN 1: Input & Preprocessing ────────────────────────┐
│ Input (str | list[str])  +  engine_type                                         │
│        │                                                                        │
│        ▼                                                                        │
│  AspectAnalyzer (Facade)  ──►  TextPreprocessor  ──►  clean text                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────── BAGIAN 2: Core Engine ──────────────────────────────┐
│  BaseEngine (Strategy interface)                                                │
│    ├── RuleBasedEngine                                                          │
│    ├── NaiveBayesEngine                                                         │
│    ├── SVMEngine                                                                │
│    └── IndoBERTEngine ──► ModelManager (Lazy Loading) ◄── Hugging Face Hub      │
│  Output: label sentimen (Positif/Negatif/Netral) + probabilitas per aspek       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────── BAGIAN 3: Output & Utility ─────────────────────────────┐
│  Evaluator (Akurasi, Presisi, Recall, F1, waktu)   Logger (error handling)      │
│  Output akhir: Aspek, Sentimen, Confidence Score (dict / DataFrame)             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Perbandingan mekanisme engine (sesuai Tabel 2.1 proposal)

| Engine | Ekstraksi / Input Format | Penentuan Sentimen (Output) |
|---|---|---|
| Rule-Based | Dependency Graph & POS Tags | Kalkulasi skor lexicon & polaritas |
| Naive Bayes | TF-IDF / N-grams | Probabilitas posterior tertinggi (argmax) |
| SVM | TF-IDF / N-grams | Pemisahan ruang vektor (hyperplane) |
| IndoBERT | Sentence pair `[CLS] Aspect [SEP] Text [SEP]` | Linear layer + Softmax |

### 7.3 Prinsip desain

| Prinsip | Penerapan |
|---|---|
| **Facade** | `AspectAnalyzer` menyembunyikan TextPreprocessor, engine, ModelManager, Evaluator, Logger. |
| **Strategy** | `BaseEngine` sebagai kontrak seragam; pergantian algoritma lewat parameter `engine`. |
| **Inheritance & Polymorphism** | Keempat engine mewarisi `BaseEngine` dan mengimplementasikan `predict()` (dan `train()` bila perlu). |
| **Lazy Loading** | Bobot IndoBERT diunduh/dimuat hanya saat pertama kali dipanggil. |
| **Fail-safe** | Semua titik risiko dibungkus try-except; kegagalan dicatat ke Logger dan sistem kembali ke safe state. |
| **Separation of concerns** | `preprocessing.py`, `engines.py`, `evaluator.py`, `analyzer.py`, dst. |

---

## 8. Functional Requirements

**Konvensi prioritas (MoSCoW):** Must = wajib pada proposal; Should = sangat disarankan; Could = opsional.
**Sumber:** "Proposal" = tertulis di proposal; **[ASUMSI]** = tambahan desain saya.

### 8.1 Modul Preprocessing (`TextPreprocessor`)

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-PRE-01 | Menerima `str` dan menghasilkan teks bersih (`clean_text`). | Must | Proposal |
| FR-PRE-02 | **Cleaning**: menghapus tanda baca, simbol, dan angka. | Must | Proposal |
| FR-PRE-03 | **Case folding**: ubah ke huruf kecil. | Must | Proposal |
| FR-PRE-04 | **Tokenisasi**: memecah teks menjadi token kata (`tokenize`). | Must | Proposal |
| FR-PRE-05 | **Spell correction** dengan jarak **Damerau-Levenshtein** (hapus, sisip, substitusi, transposisi). | Must | Proposal |
| FR-PRE-06 | **Normalisasi slang/singkatan** berbasis kamus (`slang_dict`), mis. `gmn → bagaimana`, `bgus → bagus`. | Must | Proposal |
| FR-PRE-07 | **Penghapusan stop words** (mis. "di", "dan", "yang"). | Must | Proposal |
| FR-PRE-08 | Menangani **perulangan karakter** tidak beraturan (mis. `bagusssss → bagus`). | Must | Proposal (Bab 3.1) |
| FR-PRE-09 | Urutan pipeline default: cleaning → case folding → repeated-char reduction → tokenisasi → normalisasi slang → spell correction → stop words removal. | Must | **[ASUMSI]** |
| FR-PRE-10 | **Negation-safe stop words**: kata negasi (`tidak`, `bukan`, `belum`, `jangan`, `kurang`, dsb.) **tidak boleh dihapus** karena membalik polaritas. | Must | **[ASUMSI]** |
| FR-PRE-11 | Spell correction hanya menerima koreksi bila jarak ≤ ambang (`max_edit_distance`, default 2) dan kata kandidat ada pada kamus kosakata; kata yang tidak dikenal dibiarkan. | Must | **[ASUMSI]** |
| FR-PRE-12 | Setiap tahap preprocessing dapat **diaktifkan/dinonaktifkan** lewat parameter konfigurasi. | Should | **[ASUMSI]** |
| FR-PRE-13 | Pengguna dapat menambahkan kamus slang kustom (`custom_slang`). | Could | **[ASUMSI]** |
| FR-PRE-14 | Kamus slang, stop words, dan kosakata dikemas sebagai **data resource paket** (package data), dimuat sekali dan di-cache di memori. | Must | **[ASUMSI]** |

### 8.2 Ekstraksi Aspek (Aspect Extraction)

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-ASP-01 | Mengidentifikasi **aspect term** (mis. "makanan", "pelayanan", "harga") dan **opinion term** pada teks. | Must | Proposal (Bab 2.2.1) |
| FR-ASP-02 | Library bersifat **general (lintas domain)**, tidak terikat satu domain. | Must | Proposal |
| FR-ASP-03 | Aspek default untuk eksperimen penelitian: `Aplikasi`, `Layanan`, `Harga`. | Must | Proposal (Bab 3.4) |
| FR-ASP-04 | Pengguna dapat memberikan **daftar aspek / kamus aspek** sendiri (`aspects={"Harga": ["harga","mahal","murah"], ...}`). | Should | **[ASUMSI]** |
| FR-ASP-05 | Satu teks dapat menghasilkan **beberapa aspek**, tiap aspek memiliki sentimen **independen**. | Must | Proposal (ABSA) |
| FR-ASP-06 | Untuk teks tanpa aspek terdeteksi, library mengembalikan baris dengan `aspect="UMUM"` (atau `None`) dan sentimen level-kalimat. | Should | **[ASUMSI]** |
| FR-ASP-07 | Untuk Naive Bayes/SVM, sentimen per aspek diprediksi dari **segmen teks (klausa/window)** di sekitar aspek, bukan seluruh kalimat, agar sentimen campuran tidak saling mengganggu. | Should | **[ASUMSI]** |

### 8.3 Engine Klasifikasi

#### 8.3.1 Kontrak `BaseEngine`

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-ENG-01 | `BaseEngine` adalah **kelas abstrak** (ABC) dengan atribut `logger` dan metode `train(X, y)` serta `predict(X)`. | Must | Proposal (Gambar 3.3) |
| FR-ENG-02 | Seluruh engine **wajib** mengimplementasikan `predict()` dengan format keluaran seragam. | Must | Proposal |
| FR-ENG-03 | Keluaran `predict()` seragam: label ∈ {`Positif`, `Negatif`, `Netral`} + probabilitas/confidence per aspek. | Must | Proposal |
| FR-ENG-04 | Engine baru dapat ditambahkan dengan mewarisi `BaseEngine` dan **mendaftarkan diri** pada registry engine tanpa mengubah `AspectAnalyzer`. | Should | **[ASUMSI]** |
| FR-ENG-05 | Engine yang butuh training namun belum dilatih akan menampilkan error informatif (`EngineNotFittedError`) atau memakai **model pretrained bawaan** bila tersedia. | Must | **[ASUMSI]** |

#### 8.3.2 Rule-Based

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-RB-01 | Menghitung skor sentimen berbasis **lexicon** dan aturan polaritas. | Must | Proposal |
| FR-RB-02 | Memanfaatkan **POS tags dan dependency graph** untuk mengaitkan opinion term dengan aspect term. | Must | Proposal (Tabel 2.1) |
| FR-RB-03 | Menangani **negasi** (mis. "tidak bagus" → negatif) dan **intensifier** (mis. "sangat", "banget"). | Must | **[ASUMSI]** |
| FR-RB-04 | **Fallback** ke pendekatan *window-based* (jarak token antara aspect dan opinion term) bila parser POS/dependency tidak tersedia. | Should | **[ASUMSI]** |
| FR-RB-05 | Lexicon sentimen berbahasa Indonesia dikemas dengan paket (sumber & lisensi harus jelas). | Must | **[ASUMSI]** |
| FR-RB-06 | Tidak memerlukan training; dapat langsung dipakai setelah instalasi. | Must | **[ASUMSI]** |

#### 8.3.3 Naive Bayes dan SVM

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-ML-01 | Representasi fitur **TF-IDF / N-grams** (default unigram+bigram; dapat dikonfigurasi). | Must | Proposal |
| FR-ML-02 | `NaiveBayesEngine`: klasifikasi dengan **probabilitas posterior tertinggi (argmax)**. | Must | Proposal |
| FR-ML-03 | `SVMEngine`: klasifikasi dengan **hyperplane**; confidence dihasilkan lewat kalibrasi probabilitas (mis. `probability=True` / `CalibratedClassifierCV`). | Must | Proposal + **[ASUMSI]** |
| FR-ML-04 | Mendukung `train(X, y)` pada dataset pengguna. | Must | Proposal |
| FR-ML-05 | Menyediakan **model pretrained bawaan** (dilatih pada dataset penelitian) agar `analyze()` langsung bisa dipakai tanpa training. | Should | **[ASUMSI]** |
| FR-ML-06 | Implementasi berbasis `scikit-learn` (`MultinomialNB`/`ComplementNB`, `LinearSVC`/`SVC`). | Must | **[ASUMSI]** |

#### 8.3.4 IndoBERT

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-BERT-01 | Input berformat **sentence pair**: `[CLS] Aspect [SEP] Text [SEP]`. | Must | Proposal |
| FR-BERT-02 | Klasifikasi melalui **linear layer + softmax**; keluaran berupa label + probabilitas. | Must | Proposal |
| FR-BERT-03 | Model dimuat dari **Hugging Face Hub** melalui `ModelManager` dengan **Lazy Loading** (hanya saat pertama dipanggil). | Must | Proposal |
| FR-BERT-04 | Mendukung **fine-tuning** pada dataset berlabel (`train()`), diuji pada data uji. | Must | Proposal (Tabel 3.1) |
| FR-BERT-05 | `ModelManager` menyediakan `load_model()` dan `is_loaded()`; model tidak dimuat ulang bila sudah ada di memori (cache). | Must | Proposal (Gambar 3.3) |
| FR-BERT-06 | Bila unduhan gagal (mis. koneksi putus), `ModelManager` menangkap exception, mencatat ke Logger, dan menjaga sistem pada **safe state** (tidak crash, tidak meninggalkan file setengah jadi yang merusak cache). | Must | Proposal (Bab 3.3) |
| FR-BERT-07 | Otomatis memakai **GPU bila tersedia**, jika tidak maka CPU. | Should | **[ASUMSI]** |
| FR-BERT-08 | Dependensi berat (`torch`, `transformers`) diinstal sebagai **optional extra**: `pip install absa-library[indobert]`; bila belum terpasang, error yang muncul informatif (menyarankan perintah instalasi). | Should | **[ASUMSI]** |
| FR-BERT-09 | Nama checkpoint dapat dikonfigurasi (`model_name`); kandidat default: `indolem/indobert-base-uncased` atau `indobenchmark/indobert-base-p1`. | Should | **[ASUMSI]** |
| FR-BERT-10 | Inferensi batch dengan `batch_size` dapat diatur. | Should | **[ASUMSI]** |

### 8.4 Facade API (`AspectAnalyzer`)

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-API-01 | Konstruktor menerima parameter pemilihan engine: `'rule-based'`, `'naive-bayes'`, `'svm'`, `'indobert'`. | Must | Proposal |
| FR-API-02 | `analyze(texts)` menerima `str` **atau** `list[str]`, menjalankan preprocessing + engine, dan mengembalikan hasil terstruktur. | Must | Proposal |
| FR-API-03 | Hasil `analyze()` berupa `list[dict]` atau `pandas.DataFrame` dengan kolom: `text`, `aspect`, `sentiment`, `confidence_score`. | Must | Proposal |
| FR-API-04 | `evaluate(x_test, y_test)` mengembalikan `dict` metrik (akurasi, presisi, recall, F1-score). | Must | Proposal |
| FR-API-05 | Pengguna **tidak perlu** memanggil `TextPreprocessor` atau `BaseEngine` langsung (Facade). | Must | Proposal |
| FR-API-06 | **Pergantian engine saat runtime** dapat dilakukan tanpa membuat ulang objek (mis. `analyzer.set_engine("indobert")`). | Should | Proposal (runtime switch) + **[ASUMSI]** untuk nama metode |
| FR-API-07 | Parameter `output_format` ∈ {`"dict"` (default), `"dataframe"`}. | Should | **[ASUMSI]** |
| FR-API-08 | Menyediakan `fit(X, y)`/`train` untuk melatih engine pada data pengguna. | Should | **[ASUMSI]** |
| FR-API-09 | Menyediakan `save(path)` / `load(path)` untuk persistensi model terlatih. | Should | **[ASUMSI]** |
| FR-API-10 | Menyediakan `compare_engines(x_test, y_test, engines=[...])` untuk membandingkan akurasi **dan** waktu eksekusi seluruh engine sekaligus. | Could | **[ASUMSI]** |
| FR-API-11 | Menyediakan `__version__` dan `__repr__` yang informatif. | Could | **[ASUMSI]** |

### 8.5 Evaluator

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-EVA-01 | `calculate_metrics(y_true, y_pred)` mengembalikan **Akurasi, Presisi, Recall, F1-Score** dari Confusion Matrix. | Must | Proposal |
| FR-EVA-02 | `save_report(path)` menyimpan laporan evaluasi. | Must | Proposal (Gambar 3.3) |
| FR-EVA-03 | Mengukur **waktu eksekusi** per engine (preprocessing → prediksi). | Must | Proposal |
| FR-EVA-04 | Mengembalikan **confusion matrix**. | Should | **[ASUMSI]** |
| FR-EVA-05 | Metrik per kelas dan rata-rata (macro/weighted) tersedia. | Should | **[ASUMSI]** |
| FR-EVA-06 | Format laporan: JSON dan/atau CSV/Markdown. | Could | **[ASUMSI]** |

### 8.6 Logger, Validasi Input, dan Error Handling

| ID | Requirement | Prioritas | Sumber |
|---|---|---|---|
| FR-ERR-01 | Kelas `Logger` dengan `info(message)` dan `error(message)`. | Must | Proposal (Gambar 3.3) |
| FR-ERR-02 | **Input validation**: input kosong, `None`, angka, tipe salah, atau teks ekstrem (sangat panjang/hanya simbol) ditolak dengan **pesan error informatif** tanpa crash. | Must | Proposal (Bab 3.3) |
| FR-ERR-03 | Penanganan error dengan blok **try-except** pada titik berisiko (unduh model, pemrosesan engine). | Must | Proposal |
| FR-ERR-04 | **Safe state**: setelah error, objek `AspectAnalyzer` tetap dapat dipakai untuk pemanggilan berikutnya. | Must | Proposal (Roll Back) |
| FR-ERR-05 | Hierarki exception kustom: `AbsaError` → `InvalidInputError`, `EngineNotFittedError`, `ModelLoadError`, `UnsupportedEngineError`. | Should | **[ASUMSI]** |
| FR-ERR-06 | Pada input batch yang sebagian invalid, elemen valid tetap diproses dan elemen invalid dilaporkan (tidak membatalkan seluruh batch) — perilaku dapat diatur (`on_error="skip"|"raise"`). | Should | **[ASUMSI]** |
| FR-ERR-07 | Logger memakai modul `logging` standar Python sehingga level & handler dapat dikonfigurasi pengguna. | Should | **[ASUMSI]** |
| FR-ERR-08 | Engine tidak dikenal → `UnsupportedEngineError` dengan daftar engine yang valid. | Must | **[ASUMSI]** |

---

## 9. Non-Functional Requirements

| ID | Kategori | Requirement | Target |
|---|---|---|---|
| NFR-01 | **Usability** | API mirip scikit-learn/pandas; use case dasar ≤ 5 baris kode. | M4 |
| NFR-02 | **Modularitas** | Pemisahan modul `preprocessing`, `engines`, `evaluator`, `analyzer`, `logger`, `model_manager`. | Review kode |
| NFR-03 | **Reliability** | Tidak ada crash pada input invalid; safe state setelah error. | M5, M6 |
| NFR-04 | **Testability / Quality** | Code coverage unit test | **> 80%** |
| NFR-05 | **Efisiensi memori** | Lazy Loading IndoBERT; `import absa_library` tidak memuat `torch`/`transformers`. | M8 |
| NFR-06 | **Performa** | Waktu eksekusi tiap engine diukur dan dilaporkan; tidak ada target absolut dari proposal (lihat OQ-9). | Bab 4 |
| NFR-07 | **Portabilitas** | Berjalan di Windows, Linux, macOS, dan Google Colab. | CI matrix **[ASUMSI]** |
| NFR-08 | **Kompatibilitas Python** | Python ≥ 3.9 **[ASUMSI]** | CI matrix |
| NFR-09 | **Maintainability** | Type hints, docstring (gaya NumPy/Google), format kode (black/ruff). | Lint check **[ASUMSI]** |
| NFR-10 | **Dokumentasi** | README, docs/, docstring publik 100%, minimal 1 Notebook contoh per engine. | **[ASUMSI]** |
| NFR-11 | **Reproducibility** | `random_state` tetap pada split data dan training; versi dependensi dikunci pada lingkungan eksperimen. | **[ASUMSI]** |
| NFR-12 | **Keamanan** | Validasi input; tidak mengeksekusi kode dari input; pickle model hanya dimuat dari path yang diberikan pengguna (peringatan di dokumentasi). | **[ASUMSI]** |
| NFR-13 | **Lisensi** | Open-source dengan lisensi jelas (mis. MIT/Apache-2.0); lisensi dataset & lexicon diverifikasi. | **[ASUMSI]** |
| NFR-14 | **Ukuran paket** | Wheel tanpa bobot IndoBERT (diunduh on-demand); ukuran paket tetap ringan. | **[ASUMSI]** |

---

## 10. Spesifikasi API Publik

> Nama parameter pada proposal tidak konsisten: Gambar 3.3 menulis `engine_type`, sedangkan contoh kode memakai `engine='svm'`. Dokumen ini menetapkan **`engine`** sebagai nama parameter utama dan **`engine_type`** sebagai alias (lihat OQ-3).

### 10.1 `AspectAnalyzer`

```python
class AspectAnalyzer:
    def __init__(
        self,
        engine: str = "rule-based",          # 'rule-based' | 'naive-bayes' | 'svm' | 'indobert'
        aspects: dict[str, list[str]] | None = None,
        preprocess: bool | dict = True,
        output_format: str = "dict",         # 'dict' | 'dataframe'
        on_error: str = "raise",             # 'raise' | 'skip'
        **engine_kwargs,
    ) -> None: ...

    def analyze(self, texts: str | list[str]) -> list[dict] | "pd.DataFrame": ...

    def evaluate(self, x_test: list[str], y_test: list) -> dict: ...

    def fit(self, X: list[str], y: list) -> "AspectAnalyzer": ...      # [ASUMSI]

    def set_engine(self, engine: str, **engine_kwargs) -> None: ...    # [ASUMSI] runtime switch

    def save(self, path: str) -> None: ...                             # [ASUMSI]

    @classmethod
    def load(cls, path: str) -> "AspectAnalyzer": ...                  # [ASUMSI]
```

### 10.2 Skema keluaran `analyze()`

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `text` | `str` | Teks asli masukan pengguna (sebelum preprocessing) |
| `aspect` | `str` | Aspek yang terdeteksi |
| `sentiment` | `str` | `Positif` / `Negatif` / `Netral` |
| `confidence_score` | `float` | Probabilitas keyakinan model, rentang 0.0–1.0 |

**Contoh keluaran** (satu teks, dua aspek):

```python
[
  {"text": "Barangnya bagus banget, tapi kurirnya lama",
   "aspect": "Produk",  "sentiment": "Positif", "confidence_score": 0.91},
  {"text": "Barangnya bagus banget, tapi kurirnya lama",
   "aspect": "Layanan", "sentiment": "Negatif", "confidence_score": 0.84},
]
```

### 10.3 Contoh penggunaan

```python
# 1. Instalasi (Jupyter / Colab)
# !pip install absa-library

# 2. Penggunaan dasar
from absa_library import AspectAnalyzer

analyzer = AspectAnalyzer(engine="svm")
hasil = analyzer.analyze(["Barangnya bagus banget, tapi kurirnya lama"])
print(hasil)

# 3. Ganti engine saat runtime
analyzer.set_engine("indobert")          # model diunduh saat pertama kali dipakai (lazy loading)
hasil_bert = analyzer.analyze("Aplikasinya sering error, tp harganya murah")

# 4. Evaluasi
metrik = analyzer.evaluate(x_test, y_test)
# {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1_score': ...}
```

### 10.4 Alur data `analyze()`

```
analyze(texts)
  ├─ validate_input(texts)                  → InvalidInputError bila tidak valid
  ├─ normalize ke list[str]
  ├─ TextPreprocessor.process(text)         → clean text
  ├─ extract aspects (+ opinion terms)
  ├─ engine.predict(X)                      → label + probabilitas
  ├─ Logger.info(...)
  └─ format hasil (dict / DataFrame)
```

---

## 11. Desain Kelas dan Struktur Proyek

### 11.1 Struktur direktori (sesuai Gambar 3.2 proposal)

```
absa-library/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── src/
│   └── absa_library/
│       ├── __init__.py              # ekspor AspectAnalyzer, __version__
│       ├── analyzer.py              # AspectAnalyzer (Facade)
│       ├── preprocessing.py         # TextPreprocessor + Damerau-Levenshtein
│       ├── engines.py               # BaseEngine + 4 engine (atau paket engines/)
│       ├── model_manager.py         # ModelManager (Lazy Loading)
│       ├── evaluator.py             # Evaluator
│       ├── logger.py                # Logger
│       ├── exceptions.py            # hierarki exception kustom
│       └── data/                    # kamus slang, stop words, kosakata, lexicon, model pretrained
├── tests/
│   ├── test_preprocessing.py
│   ├── test_engines.py
│   ├── test_evaluator.py
│   ├── test_analyzer.py
│   ├── test_validation.py
│   └── test_error_handling.py
├── docs/
├── notebooks/
│   ├── 01_quickstart.ipynb
│   └── 02_compare_engines.ipynb
└── experiments/                     # skrip eksperimen Bab 4 (di luar paket)
```

> Proposal menyebut `preprocessing.py`, `engines.py`, `evaluator.py`. Jika `engines.py` terlalu besar, boleh dipecah menjadi paket `engines/` tanpa mengubah API publik.

### 11.2 Class diagram (ringkas, sesuai Gambar 3.3)

| Kelas | Atribut | Metode | Peran |
|---|---|---|---|
| `AspectAnalyzer` | `-engine: BaseEngine`, `-preprocessor: TextPreprocessor`, `-evaluator: Evaluator` | `__init__(engine_type)`, `analyze(texts)`, `evaluate(X_test, y_test)` | **Facade** |
| `TextPreprocessor` | `-slang_dict: Dict` | `clean_text()`, `normalize()`, `tokenize()`, `spell_correction()` | Preprocessing |
| `BaseEngine` (abstract) | `#logger: Logger` | `train(X, y)`, `predict(X)` | **Strategy interface** |
| `RuleBasedEngine` | – | `predict(X)` | Concrete Strategy |
| `NaiveBayesEngine` | – | `train()`, `predict()` | Concrete Strategy |
| `SVMEngine` | – | `train()`, `predict()` | Concrete Strategy |
| `IndoBERTEngine` | `-model_manager: ModelManager` | `train()`, `predict()` | Concrete Strategy |
| `ModelManager` | – | `load_model()`, `is_loaded()` | **Lazy Loading** |
| `Evaluator` | – | `calculate_metrics(y_true, y_pred)`, `save_report(path)` | Evaluasi |
| `Logger` | – | `info(message)`, `error(message)` | Logging |

**Relasi:**
`AspectAnalyzer` ◆— `TextPreprocessor` (Facade/composition) · `AspectAnalyzer` ··> `Evaluator` (uses) · `AspectAnalyzer` → `BaseEngine` (delegates, Strategy) · 4 engine ▷ `BaseEngine` (inherits) · `IndoBERTEngine` ◆— `ModelManager` (lazy loading) · `BaseEngine` ··> `Logger` (logs).

### 11.3 Registry engine **[ASUMSI]**

```python
_ENGINE_REGISTRY = {
    "rule-based":  RuleBasedEngine,
    "naive-bayes": NaiveBayesEngine,
    "svm":         SVMEngine,
    "indobert":    IndoBERTEngine,   # import torch/transformers dilakukan di dalam kelas (lazy)
}
```

Alias yang diterima: `rule_based`, `nb`, `naivebayes`, `bert` → dinormalisasi menjadi nama kanonik.

---

## 12. Spesifikasi Data dan Dataset

### 12.1 Sumber data (sesuai Bab 3.4 proposal)

| Sumber | Deskripsi | Catatan |
|---|---|---|
| **Dataset publik** (Kaggle / Hugging Face) | Ulasan e-commerce berbahasa Indonesia (mis. Tokopedia/Shopee) | Memudahkan replikasi; verifikasi lisensi |
| **Web scraping** | Ulasan Google Play Store aplikasi layanan publik/transportasi (MyPertamina, KAI Access, Traveloka) | Data informal sangat beragam; patuhi ToS, anonimkan data pribadi |

### 12.2 Pelabelan

- Label **aspek**: `Aplikasi`, `Layanan`, `Harga` (untuk dataset eksperimen).
- Label **sentimen**: `Positif`, `Negatif`, `Netral`.
- Metode: **manual annotation** dan/atau bantuan kamus leksikon sentimen.
- **[ASUMSI]** Panduan anotasi tertulis; bila memungkinkan 2 anotator pada sampel (mis. 10–20%) untuk menghitung inter-annotator agreement (Cohen's κ).

### 12.3 Pembagian data

| Bagian | Proporsi | Kegunaan |
|---|---|---|
| Data latih | **80%** | Training NB, SVM, fine-tuning IndoBERT |
| Data uji | **20%** | Evaluasi seluruh engine (termasuk Rule-Based) |

**[ASUMSI]** Gunakan *stratified split* dengan `random_state` tetap. Bila perlu *hyperparameter tuning*, ambil data validasi dari dalam 80% data latih (cross-validation), **bukan** dari data uji.

### 12.4 Skema dataset berlabel **[ASUMSI]**

| Kolom | Tipe | Contoh |
|---|---|---|
| `id` | int | 1024 |
| `text` | str | "aplikasinya sering error tp harganya murah" |
| `aspect` | str | `Aplikasi` |
| `sentiment` | str | `Negatif` |
| `source` | str | `playstore_kai_access` |

(Satu teks dengan banyak aspek = banyak baris.)

### 12.5 Aspek etika dan privasi **[ASUMSI]**

- Hapus/anonimkan identitas (nama pengguna, nomor telepon, email) pada data hasil scraping.
- Cantumkan sumber, tanggal pengambilan, dan lisensi data pada dokumentasi.

---

## 13. Strategi Pengujian dan Evaluasi

### 13.1 Pengujian perangkat lunak (pytest)

| Komponen | Contoh test case | Target |
|---|---|---|
| `preprocessing.py` | `bgus → bagus`; tanda baca/angka terhapus; `bagusssss → bagus`; negasi tidak terhapus; spell correction Damerau-Levenshtein (termasuk transposisi, mis. `bgaus → bagus`) | Coverage tinggi |
| `engines.py` | Tiap engine: `train()` (bila perlu) dan `predict()` menghasilkan format yang benar; label ∈ {Positif, Negatif, Netral}; confidence ∈ [0,1]; engine belum dilatih → `EngineNotFittedError` | Menggunakan data dummy |
| `evaluator.py` | Nilai akurasi/presisi/recall/F1 sesuai perhitungan manual pada data dummy | Akurat |
| `analyzer.py` | `analyze()` menerima `str` & `list`; format output sesuai; `set_engine()` bekerja; `UnsupportedEngineError` | Fungsional |
| Input validation | Input kosong `""`, `None`, `123`, `[]`, list campuran, teks sangat panjang, hanya simbol → error informatif / tanpa crash | Tabel 3.1 (Security) |
| Error handling | Simulasi gagal unduh IndoBERT (mock `from_pretrained` raise exception) → tercatat di Logger, sistem tetap hidup | Tabel 3.1 (Roll Back) |
| Lazy loading | `import absa_library` & engine non-BERT tidak memicu import/unduh `transformers`; `ModelManager.is_loaded()` false sebelum pemanggilan pertama | M8 |
| IndoBERT | Gunakan **mock/model kecil** di CI agar test cepat dan tidak bergantung jaringan | Praktis |

**Perintah:**

```bash
pytest --cov=absa_library --cov-report=term-missing --cov-fail-under=80
```

### 13.2 Tabel skenario pengujian (sesuai Tabel 3.1)

| Komponen | Metode | Target |
|---|---|---|
| Kode perangkat lunak (preprocessing, engines, evaluator) | Unit testing pytest | Coverage > 80%, bebas error fatal |
| Keamanan input | Input Validation Testing | Input invalid ditolak aman + pesan error |
| Pemulihan kesalahan | Error Handling & Exception Testing | Kembali ke safe state, tercatat di Logger |
| Engine Rule-Based | Evaluasi data uji | Akurasi, Presisi, Recall, F1, Waktu |
| Engine Naive Bayes | Training & testing | Akurasi, Presisi, Recall, F1, Waktu |
| Engine SVM | Training & testing | Akurasi, Presisi, Recall, F1, Waktu |
| Engine IndoBERT | Fine-tuning & testing | Akurasi, Presisi, Recall, F1, Waktu |

### 13.3 Evaluasi model (data mining)

- Metrik dari **Confusion Matrix**: Akurasi, Presisi, Recall, F1-Score.
- **Waktu eksekusi**: diukur dari preprocessing hingga prediksi. **[ASUMSI]** Laporkan juga waktu training dan waktu *cold start* (termasuk unduh/muat IndoBERT) terpisah dari waktu inferensi *warm*; jalankan pengukuran berulang (mis. 5 kali) dan laporkan rata-rata ± simpangan baku; catat spesifikasi hardware (CPU/GPU).
- **[ASUMSI]** Laporkan **macro-F1** selain akurasi karena kelas sentimen biasanya tidak seimbang.
- Hasil disajikan pada **Bab 4** bersama demonstrasi integrasi library.

---

## 14. Packaging, Distribusi, dan Dokumentasi

### 14.1 Langkah (sesuai Gambar 3.2 proposal)

1. **Setup environment** — `venv`/`conda`; install pandas, scikit-learn, transformers, pytest.
2. **Inisialisasi struktur proyek** — `src/`, `tests/`, `docs/`, `pyproject.toml`.
3. **Implementasi kode modular** — `preprocessing.py`, `engines.py`, `evaluator.py`.
4. **Perancangan API** — kelas `AspectAnalyzer`.
5. **Unit testing** — pytest + code coverage > 80%.
6. **Gerbang kualitas** — jika belum lulus/coverage ≤ 80% → perbaiki kode (loop kembali ke langkah 3).
7. **Build & packaging** — `python -m build` → `.whl` dan `.tar.gz`.
8. **Distribusi ke PyPI** — upload paket; hasil akhir: `pip install nama-library`.

### 14.2 Contoh `pyproject.toml` **[ASUMSI]**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "absa-library"            # verifikasi ketersediaan nama di PyPI
version = "0.1.0"
description = "Aspect-Based Sentiment Analysis untuk teks Bahasa Indonesia informal (multi-engine)"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT" }
authors = [{ name = "Yusuf Rahmadhani Asy'Ari" }]
dependencies = [
    "pandas",
    "scikit-learn",
    "numpy",
]

[project.optional-dependencies]
indobert = ["torch", "transformers"]
rule     = ["stanza"]             # bila memakai dependency parser/POS (lihat OQ-4)
dev      = ["pytest", "pytest-cov", "ruff", "black", "build", "twine"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
absa_library = ["data/*"]

[tool.pytest.ini_options]
addopts = "--cov=absa_library --cov-fail-under=80"
```

### 14.3 Dokumentasi minimum **[ASUMSI]**

- `README.md`: deskripsi, instalasi, quickstart, tabel engine, lisensi.
- `docs/`: panduan pengguna, referensi API, penjelasan arsitektur (Facade/Strategy), panduan kontribusi, cara menambah engine baru.
- `notebooks/`: quickstart dan perbandingan engine (siap jalan di Google Colab).

---

## 15. Dependensi dan Lingkungan

| Kebutuhan | Pustaka | Keterangan |
|---|---|---|
| Data & output | `pandas`, `numpy` | Output DataFrame |
| ML klasik | `scikit-learn` | TF-IDF, NB, SVM, metrik |
| Deep learning | `torch`, `transformers` | Optional extra `[indobert]`; Lazy Loading |
| POS/dependency (Rule-Based) | `stanza` / `spaCy` (kandidat) | Optional; ada fallback window-based |
| Pengujian | `pytest`, `pytest-cov`, `unittest.mock` | Coverage > 80% |
| Packaging | `build`, `setuptools`, `twine` | `.whl` + `.tar.gz` → PyPI |
| Kualitas kode | `ruff`/`black`, `mypy` (opsional) | **[ASUMSI]** |
| Lingkungan pengembangan | Jupyter Notebook / Google Colab, venv/conda | Sesuai proposal |
| Eksperimen IndoBERT | GPU (mis. Colab T4) disarankan untuk fine-tuning | **[ASUMSI]** |

---

## 16. Milestone dan Rencana Pengerjaan

> Proposal tidak mencantumkan jadwal. Pembagian fase di bawah adalah **usulan** mengikuti SDLC pada Bab 2.2.4 dan alur Gambar 3.2. Durasi disesuaikan dengan kalender proyek akhir.

| Fase | Kegiatan | Keluaran | Estimasi (usulan) |
|---|---|---|---|
| **M0 — Requirement** | Finalisasi PRD, keputusan Open Questions | PRD v1.0 disetujui | 1 minggu |
| **M1 — Setup & Desain** | Environment, struktur proyek, `pyproject.toml`, skeleton kelas, CI dasar | Repo siap, skeleton `BaseEngine`/`AspectAnalyzer` | 1 minggu |
| **M2 — Data** | Pengumpulan (dataset publik + scraping), filtering, pelabelan, split 80/20 | Dataset berlabel + panduan anotasi | 2–3 minggu |
| **M3 — Preprocessing** | Cleaning, normalisasi slang, Damerau-Levenshtein, stop words (negation-safe) + unit test | `preprocessing.py` teruji | 1–2 minggu |
| **M4 — Engine klasik** | Rule-Based, Naive Bayes, SVM + training + unit test | 3 engine berfungsi | 2 minggu |
| **M5 — IndoBERT** | `ModelManager` lazy loading, fine-tuning, error handling unduh + unit test (mock) | `IndoBERTEngine` berfungsi | 2 minggu |
| **M6 — Facade & Utility** | `AspectAnalyzer`, `Evaluator`, `Logger`, validasi input, exceptions | API lengkap | 1–2 minggu |
| **M7 — Testing** | Lengkapi unit test, coverage > 80%, test validasi & roll back | Laporan coverage | 1 minggu |
| **M8 — Eksperimen (Bab 4)** | Evaluasi 4 engine (metrik + waktu), demonstrasi Notebook | Tabel/grafik hasil | 1–2 minggu |
| **M9 — Packaging & Rilis** | `python -m build`, TestPyPI → PyPI, dokumentasi | Paket `pip install` tersedia | 1 minggu |
| **M10 — Penulisan Buku** | Bab 4–5, revisi, sidang | Buku proyek akhir | sesuai kalender |

---

## 17. Risiko dan Mitigasi

| ID | Risiko | Dampak | Prob. | Mitigasi |
|---|---|---|---|---|
| R1 | **Ketersediaan parser POS/dependency Bahasa Indonesia** yang ringan untuk Rule-Based | Rule-Based tidak sesuai desain Tabel 2.1 | Sedang | Evaluasi Stanza/spaCy; sediakan fallback window-based; jadikan dependensi optional |
| R2 | **Ukuran & waktu unduh IndoBERT**, fine-tuning berat di CPU | Eksperimen lambat; UX buruk | Tinggi | Lazy loading + cache; optional extra; gunakan GPU Colab; sediakan checkpoint hasil fine-tune di Hugging Face Hub |
| R3 | **Gangguan jaringan / Hugging Face Hub** saat unduh | IndoBERT gagal dimuat | Sedang | try-except + Logger + safe state; pesan error jelas; dukung `local_files_only`/path lokal |
| R4 | **Stop words menghapus negasi** | Polaritas terbalik, akurasi turun | Tinggi | FR-PRE-10 (negation-safe), uji khusus negasi |
| R5 | **Spell correction salah mengoreksi** (kata gaul/nama merek jadi kata lain) | Noise baru | Sedang | Ambang jarak edit, kamus kosakata, urutan normalisasi slang sebelum spell correction, opsi nonaktif |
| R6 | **Kualitas dan konsistensi pelabelan** rendah | Evaluasi tidak valid | Sedang | Panduan anotasi, double-annotate sampel, hitung kappa |
| R7 | **Ketidakseimbangan kelas** (dominan positif) | Akurasi menyesatkan | Tinggi | Laporkan macro-F1, class weights, stratified split |
| R8 | **Ukuran dataset kecil** → IndoBERT overfit | Hasil tidak stabil | Sedang | Early stopping, validasi silang, augmentasi ringan |
| R9 | **Coverage > 80% sulit untuk kode IndoBERT** (butuh model besar) | Target tidak tercapai | Sedang | Mock model, tiny model untuk test, pisahkan logika murni dari I/O |
| R10 | **Lisensi** dataset/lexicon/model membatasi redistribusi | Tidak bisa dikemas/dipublikasikan | Sedang | Verifikasi lisensi sejak awal; unduh on-demand bila tidak boleh dibundel |
| R11 | **Scraping Play Store** melanggar ToS/berubah struktur | Data tidak tersedia | Sedang | Prioritaskan dataset publik; gunakan library scraper resmi/legal; anonimkan data |
| R12 | **Nama paket PyPI sudah dipakai** | Gagal publish | Rendah | Cek lebih awal; siapkan nama alternatif; gunakan TestPyPI |
| R13 | **Konflik dependensi** (torch/transformers/numpy) | Instalasi gagal | Sedang | Batas versi minimal, CI matrix, optional extras |
| R14 | **Jadwal proyek akhir** terbatas | Fitur tidak selesai | Sedang | Prioritas MoSCoW; tunda fitur "Could" |

---

## 18. Open Questions dan Asumsi

Butir berikut **perlu dikonfirmasi dengan pembimbing / diputuskan penulis** sebelum implementasi.

| ID | Pertanyaan / Ketidakkonsistenan | Usulan Default dalam PRD |
|---|---|---|
| OQ-1 | **Nama final library** & ketersediaannya di PyPI (proposal memakai placeholder `absa-library` / `nama-library`). | `absa-library` / `absa_library` |
| OQ-2 | **Definisi tugas ABSA**: apakah aspek dideteksi otomatis oleh library, atau diberikan pengguna? Proposal menyebut ekstraksi aspek (Bab 2.2.1) tetapi IndoBERT memakai *sentence pair* `[CLS] Aspect [SEP] Text` yang membutuhkan aspek sebagai input. | Pipeline 2 tahap: (1) deteksi aspek via kamus aspek (bisa dikustomisasi), (2) klasifikasi sentimen per aspek |
| OQ-3 | **Nama parameter**: `engine_type` (diagram) vs `engine` (contoh kode). | `engine` utama, `engine_type` alias |
| OQ-4 | **Rule-Based** memakai *Dependency Graph & POS Tags* (Tabel 2.1): pustaka apa yang dipakai (Stanza/spaCy/lainnya)? Apakah boleh ada fallback? | Stanza optional + fallback window-based |
| OQ-5 | **Lexicon sentimen** Indonesia yang dipakai (mis. InSet) dan lisensinya. | Verifikasi lisensi; bundel bila diizinkan, jika tidak unduh on-demand |
| OQ-6 | **Model pretrained bawaan** untuk NB/SVM/IndoBERT: apakah library mengirim model terlatih (dilatih pada dataset penelitian) agar `analyze()` langsung jalan, atau pengguna harus `fit()` dulu? Proposal menyatakan hambatan "melatih ulang model dari nol" sebagai masalah. | Sediakan model pretrained untuk 3 engine ML (NB/SVM kecil dibundel; IndoBERT di Hugging Face Hub) **dan** dukung `fit()` |
| OQ-7 | **Checkpoint IndoBERT**: `indolem/indobert-base-uncased` (Koto dkk. — dirujuk proposal) atau `indobenchmark/indobert-base-p1`? | `indolem/indobert-base-uncased`, dapat dikonfigurasi |
| OQ-8 | **Label Netral**: pada Bab 3.4 sentimen mencakup Netral; pastikan dataset memiliki cukup sampel kelas Netral. | 3 kelas (Positif/Negatif/Netral) |
| OQ-9 | **Target akurasi/F1 minimum** & batas waktu eksekusi? Proposal hanya menyebut "kompetitif". | Tidak ada target absolut; bandingkan antar-engine & dengan baseline |
| OQ-10 | **Ukuran dataset** minimal (jumlah ulasan/aspek) dan jumlah anotator. | ≥ 2.000–5.000 pasangan teks-aspek, 2 anotator pada sampel (**perlu disetujui**) |
| OQ-11 | **Format `output`**: `list[dict]` default atau `DataFrame` default? Proposal menyebut keduanya. | `list[dict]` default, `output_format="dataframe"` opsional |
| OQ-12 | **Metode `evaluate(x_test, y_test)`**: bentuk `y_test` — hanya label sentimen, atau pasangan (aspek, sentimen)? | Pasangan (aspek, sentimen) per baris; label sentimen saja diterima untuk mode sentence-level |
| OQ-13 | **Struktur modul**: `engines.py` tunggal atau paket `engines/`? | Mulai `engines.py`, pecah bila besar |
| OQ-14 | **Lisensi library** (MIT/Apache-2.0/GPL). | MIT |
| OQ-15 | **Versi Python minimum** dan OS yang didukung. | Python ≥ 3.9; Windows/Linux/macOS + Colab |
| OQ-16 | Typo pada proposal yang sebaiknya dibenahi di buku akhir: "polariasi" → "polarisasi"; "Netral" tertulis "Netral" vs "Netral"/"Netral" (konsistensi), "Mustakim & Priyanti" vs "Priyanta"; kalimat "pembelajaran.tersebut" di Bab 2; Bab 1.5 masih berisi teks template ("Jelaskan tentang..."). | Perbaiki saat revisi |
| OQ-17 | **Konsistensi sitasi**: sebagian rujukan di Daftar Pustaka ([5], [10]) tidak tampak dirujuk di isi; penomoran [3]/[4]/[8]/[15] pada Tabel 2.2 perlu dicek. | Rapikan saat revisi |

---

## 19. Definition of Done dan Acceptance Criteria

### 19.1 Acceptance criteria per fitur

| Fitur | Kriteria penerimaan (Given/When/Then) |
|---|---|
| **Instalasi** | *Given* Python bersih, *when* `pip install absa-library`, *then* instalasi sukses dan `from absa_library import AspectAnalyzer` berjalan tanpa error. |
| **Analisis dasar** | *Given* `AspectAnalyzer(engine="svm")`, *when* `analyze(["Barangnya bagus banget, tapi kurirnya lama"])`, *then* hasil memuat kolom `text`, `aspect`, `sentiment`, `confidence_score` dan sentimen per aspek independen. |
| **Input fleksibel** | *When* `analyze("kalimat")` atau `analyze(["k1","k2"])`, *then* keduanya valid dan format keluaran konsisten. |
| **Multi-engine** | *When* engine diset ke tiap nilai {rule-based, naive-bayes, svm, indobert}, *then* semua menghasilkan keluaran dengan skema yang sama tanpa mengubah kode pemanggil. |
| **Preprocessing** | *Given* "bgus bgt tp lamaaa!!", *then* teks bersih menghasilkan bentuk baku (mis. "bagus banget tapi lama") dan negasi tidak terhapus. |
| **Lazy loading** | *Given* engine selain IndoBERT, *then* tidak ada unduhan/import model BERT; *given* IndoBERT, *then* model dimuat pada pemanggilan pertama dan tidak dimuat ulang pada pemanggilan berikutnya. |
| **Validasi input** | *When* input `""`, `None`, `123`, atau `[]`, *then* muncul pesan error informatif, program **tidak crash**, objek tetap bisa dipakai. |
| **Roll back** | *Given* unduhan IndoBERT gagal (disimulasikan), *then* error tercatat di Logger, tidak ada crash runtime, sistem kembali ke safe state. |
| **Evaluasi** | *When* `evaluate(x_test, y_test)`, *then* mengembalikan akurasi, presisi, recall, F1 yang sama dengan perhitungan referensi (scikit-learn/hitung manual) pada data dummy. |
| **Packaging** | *When* `python -m build`, *then* terbentuk `.whl` dan `.tar.gz`; paket dapat diinstal di virtual environment baru. |

### 19.2 Definition of Done (proyek)

- [ ] Seluruh requirement **Must** terimplementasi dan lulus test.
- [ ] **Code coverage > 80%** (laporan terlampir).
- [ ] Empat engine terevaluasi pada data uji 20% (Akurasi, Presisi, Recall, F1, Waktu Eksekusi).
- [ ] Pengujian validasi input dan error handling/roll back lulus.
- [ ] Paket berhasil dibuild dan dipublikasikan (TestPyPI/PyPI) dan terinstal bersih.
- [ ] Dokumentasi (README, docs, Notebook contoh) lengkap.
- [ ] Notebook demonstrasi integrasi berjalan di Google Colab.
- [ ] Hasil dituangkan pada Bab 4 dan kesimpulan pada Bab 5.

---

## 20. Lampiran

### 20.1 Glosarium

| Istilah | Arti |
|---|---|
| **ABSA** | Aspect-Based Sentiment Analysis: analisis sentimen per aspek yang dibahas dalam teks. |
| **Aspect term / Opinion term** | Objek/fitur yang dievaluasi / ungkapan opini yang melekat pada aspek. |
| **Facade Pattern** | Pola desain yang menyediakan antarmuka tunggal untuk subsistem yang kompleks. |
| **Strategy Pattern** | Pola desain yang memungkinkan algoritma dipertukarkan saat runtime melalui interface seragam. |
| **Lazy Loading** | Penundaan pemuatan sumber daya berat hingga benar-benar dibutuhkan. |
| **Damerau-Levenshtein** | Jarak edit minimum (hapus, sisip, substitusi, transposisi) antar dua string. |
| **TF-IDF** | Pembobotan fitur teks berdasarkan frekuensi term dan kelangkaan antar dokumen. |
| **Code Coverage** | Persentase baris kode yang dieksekusi oleh unit test. |
| **Safe state** | Kondisi sistem tetap valid dan dapat dipakai setelah terjadi error. |
| **Boilerplate code** | Kode dasar berulang (preprocessing, setup) yang tidak berkaitan dengan inti analisis. |

### 20.2 Ringkasan penelitian terkait (dari proposal)

| Peneliti (Tahun) | Fokus | Keterbatasan relatif terhadap proyek ini |
|---|---|---|
| Mustakim & Priyanta (2022) | Opini aplikasi KAI Access dengan NBC & SVM | Tidak dikemas sebagai library reusable |
| Lestarie (2024) | Sentimen layanan PT Citilink Indonesia | Spesifik domain; bukan produk siap pakai |
| Hammami dkk. (2026) | ABSA aplikasi pemerintah dengan IndoBERT-CNN | Skrip monolitik di Jupyter |
| Dadap dkk. (2023) | ABSA domain berita (rule-based + BiLSTM) | Kaku terhadap arsitektur tertentu |
| Renaldi (2025) | Keluhan PDAM Surya Sembada (Cosine Similarity + Rule-Based) | Spesifik satu domain; bukan library |

### 20.3 Referensi utama (dari Daftar Pustaka proposal)

- Python Software Foundation, *Python Packaging User Guide* — packaging.python.org
- Okken, B., *Python Testing with pytest*, 2nd ed., 2022
- Sommerville, I., *Software Engineering*, 10th ed., 2015
- Koto dkk., *IndoLEM and IndoBERT*, COLING 2020
- Tripathy & Sharaff (2026); Singh dkk. (2026); Cai dkk. (2025) — kajian ABSA

---

*— Akhir dokumen PRD —*