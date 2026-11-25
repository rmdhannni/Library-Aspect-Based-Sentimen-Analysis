# Changelog

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/1.1.0/) dan
[Semantic Versioning](https://semver.org/lang/id/).

## [0.1.0] - 2026-11-21

### Ditambahkan

- `AspectAnalyzer` sebagai facade: `analyze`, `fit`, `evaluate`,
  `compare_engines`, `set_engine`, `save`, `load`.
- Preprocessing Bahasa Indonesia informal: cleaning, case folding, reduksi
  karakter berulang, normalisasi slang, koreksi ejaan Damerau-Levenshtein,
  serta stop word yang aman terhadap negasi.
- Ekstraksi aspek open-domain (`aspects="auto"`), mode kamus, dan mode tanpa
  aspek, lengkap dengan segmentasi klausa.
- Engine `rule-based` berbasis leksikon, negasi, intensifier, dan jarak aspek.
- Engine `naive-bayes` (ComplementNB) dan `svm` (LinearSVC terkalibrasi) dengan
  TF-IDF unigram+bigram serta model pretrained bawaan.
- Engine `indobert` dengan `ModelManager` lazy loading, fine-tuning, dan safe
  state saat unduhan gagal.
- `Evaluator` untuk akurasi, presisi, recall, F1 (macro/weighted), confusion
  matrix, waktu eksekusi, dan laporan JSON/CSV/Markdown.
- Hierarki exception `AbsaError`, validasi input, dan mode `on_error`.
- Data paket: kamus slang, stop words, kosakata, leksikon sentimen, negasi,
  intensifier, dan kosakata aspek.
- 294 unit test dengan cakupan 97%.
