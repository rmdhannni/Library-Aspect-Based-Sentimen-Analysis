# absa-library

Library Python untuk **Aspect-Based Sentiment Analysis (ABSA)** pada teks Bahasa
Indonesia informal: ulasan aplikasi, e-commerce, dan layanan.

Library ini menyediakan satu antarmuka (`AspectAnalyzer`) dengan empat engine
yang dapat diganti saat runtime:

| Engine | Metode | Training | Keluaran |
|---|---|---|---|
| `rule-based` | Leksikon + negasi + intensifier + jarak aspek | Tidak perlu | Label + confidence |
| `naive-bayes` | TF-IDF (unigram+bigram) + ComplementNB | Ya / model bawaan | Probabilitas posterior |
| `svm` | TF-IDF + LinearSVC terkalibrasi | Ya / model bawaan | Probabilitas terkalibrasi |
| `indobert` | Sentence pair `[CLS] aspek [SEP] teks [SEP]` | Fine-tuning (opsional) | Softmax |

## Instalasi

```bash
pip install absa-library
```

Untuk engine IndoBERT (mengunduh `torch` dan `transformers`):

```bash
pip install "absa-library[indobert]"
```

Python minimum: **3.9**. Dependensi inti: `numpy`, `pandas`, `scikit-learn`,
`joblib`.

## Quickstart

```python
from absa_library import AspectAnalyzer

analyzer = AspectAnalyzer()                     # rule-based, aspek otomatis
hasil = analyzer.analyze("Barangnya bagus banget, tapi kurirnya lama")
print(hasil)
```

```python
[
    {"text": "Barangnya bagus banget, tapi kurirnya lama",
     "aspect": "barang", "sentiment": "Positif", "confidence_score": 0.7958},
    {"text": "Barangnya bagus banget, tapi kurirnya lama",
     "aspect": "kurir",  "sentiment": "Negatif", "confidence_score": 0.6726},
]
```

Input dapat berupa satu string atau daftar string. Output dapat berupa
`list[dict]` (default) atau `pandas.DataFrame`:

```python
analyzer = AspectAnalyzer(output_format="dataframe")
df = analyzer.analyze(["aplikasinya sering error", "harganya murah"])
```

## Mode aspek

```python
# 1. Otomatis (default): aspek ditemukan dari kata benda di sekitar opini.
AspectAnalyzer(aspects="auto")

# 2. Kamus sendiri: kendali penuh atas label aspek.
AspectAnalyzer(aspects={
    "Produk":  ["barang", "produk"],
    "Layanan": ["kurir", "pengiriman", "layanan"],
    "Harga":   ["harga", "biaya", "ongkir"],
})

# 3. Tanpa aspek: sentimen level kalimat, aspek = "UMUM".
AspectAnalyzer(aspects=None)
```

Untuk teks tanpa aspek terdeteksi, hasil tetap keluar dengan `aspect="UMUM"`.

## Engine

```python
analyzer = AspectAnalyzer(engine="svm")        # alias: nb, naivebayes, bert
analyzer.set_engine("rule-based")              # ganti tanpa membuat objek baru
```

Engine `naive-bayes` dan `svm` memuat model pretrained bawaan saat pertama kali
dipakai. Model juga dapat dilatih ulang:

```python
analyzer = AspectAnalyzer(engine="naive-bayes")
analyzer.fit(X_train, y_train)                 # label: Positif/Negatif/Netral
```

`fit()` menerima label sentimen saja atau pasangan `(aspek, sentimen)`:

```python
analyzer.fit(
    ["barang bagus tapi kurir lama"],
    [("barang", "Positif"), ("kurir", "Negatif")],
)
```

## Evaluasi

```python
metrik = analyzer.evaluate(x_test, y_test)
# {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1_score': ...,
#  'per_class': {...}, 'confusion_matrix': [[...]], 'elapsed_seconds': ...}

perbandingan = analyzer.compare_engines(
    x_test, y_test,
    engines=["rule-based", "naive-bayes", "svm"],
)
```

`y_test` boleh berupa label (`"Positif"`), pasangan `("aspek", "Positif")`,
atau `{"aspect": ..., "sentiment": ...}`. Metrik presisi/recall/F1 memakai
rata-rata **macro** agar kelas minoritas tetap terlihat (R7).

Laporan dapat disimpan sebagai JSON, CSV, atau Markdown:

```python
analyzer.evaluator.save_report("laporan.md")
```

## Simpan dan muat

```python
analyzer.save("model_absa")                    # direktori
analyzer = AspectAnalyzer.load("model_absa")
```

## IndoBERT

```python
analyzer = AspectAnalyzer(engine="indobert")   # model diunduh saat analyze() pertama
analyzer = AspectAnalyzer(
    engine="indobert",
    model_name="indolem/indobert-base-uncased",  # dapat diganti
    max_length=128,
    batch_size=16,
)
```

`torch` dan `transformers` hanya diimpor saat model benar-benar dimuat. Bila
unduhan gagal, error dicatat ke logger, `ModelLoadError` dilempar, dan objek
tetap berada pada kondisi aman (tidak ada file setengah jadi yang dipakai).

Fine-tuning:

```python
analyzer = AspectAnalyzer(engine="indobert", epochs=3, learning_rate=2e-5)
analyzer.fit(X_train, y_train)
```

## Validasi dan penanganan error

```python
AspectAnalyzer(on_error="raise")   # default: error langsung dilempar
AspectAnalyzer(on_error="skip")    # item invalid dilewati, batch tetap jalan
```

Input yang ditolak dengan pesan jelas: `None`, angka, string kosong, hanya
simbol/angka, teks > 10.000 karakter, dan batch > 1.000 teks. Semua error
mewarisi `AbsaError`.

## Atribusi data

Kode library berlisensi MIT. Beberapa berkas data berasal dari sumber pihak
ketiga dengan lisensi masing-masing (MIT, Apache-2.0, CC BY 4.0). Rincian ada
di berkas [`NOTICE`](NOTICE).

## Pengembangan

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -e ".[dev]"
pytest                            # coverage minimal 80%
ruff check src tests experiments
python -m build
```

Aset paket dibangun ulang dari data mentah (tidak di-commit) dengan:

```bash
python experiments/build_package_data.py
python experiments/train_pretrained.py
```

## Lisensi

MIT. Lihat [`LICENSE`](LICENSE).
