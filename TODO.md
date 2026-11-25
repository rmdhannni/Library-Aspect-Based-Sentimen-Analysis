# TODO

## Selesai
- [x] Aktifkan skill caveman otomatis di folder ini lewat `AGENTS.md`
- [x] Verifikasi `AGENTS.md` tidak di-ignore git
- [x] Tetapkan struktur memori: satu file `memory/MEMORY.md`
- [x] Tambah aturan baca/tulis memori ke `AGENTS.md`
- [x] Isi entri sesi pertama di `memory/MEMORY.md`
- [x] Bersihkan `Data/`: hapus Tokopedia, repo `.git` bersarang, Sastrawi root words, dan aset tak terpakai
- [x] Susun ulang `Data/raw/{ai_chatbots_reviews.csv, prdect_id.csv, lexicon/}` + `Data/processed/`
- [x] Inisialisasi git, `.gitignore`, `.venv`, dan instalasi editable
- [x] T1 — skeleton paket `src/absa_library`, `pyproject.toml`, exception, logger, resources, types
- [x] T2 — `preprocessing.py`: cleaning, case folding, reduksi karakter, slang, Damerau-Levenshtein, stop word aman negasi
- [x] T3 — `aspects.py`: ekstraksi otomatis, mode kamus, mode none, segmentasi klausa
- [x] T4 — `engines/rule_based.py`: leksikon, negasi, intensifier, jarak aspek
- [x] T5 — `engines/naive_bayes.py` + `engines/svm.py` + registry + model pretrained bawaan
- [x] T6 — `engines/indobert.py` + `model_manager.py` dengan lazy loading, fine-tuning, dan safe state
- [x] T7 — `analyzer.py`, `evaluator.py`, persistensi, `compare_engines`, validasi input
- [x] Data paket: kamus slang, stop word, kosakata, leksikon, negasi, intensifier, kosakata aspek
- [x] Unit test 294 kasus, coverage 97.22%
- [x] `ruff check` bersih
- [x] `python -m build` + `twine check` lulus
- [x] Instal wheel di venv bersih dan jalankan tiga engine non-BERT
- [x] README, CHANGELOG, dan NOTICE atribusi data
- [x] Susun rencana 80% sidang progress dan definisi MVP satu engine
- [x] Tetapkan evaluasi dua lapis dan gold set 500 baris
- [x] Buat `demo_absa.py` dan verifikasi rule-based, NB, dan SVM
- [x] Identifikasi bug generalitas koreksi ejaan dan bias token NB
- [x] Commit awal `0f23da9` dan push branch `main` + tag `v0.1.0` ke GitHub

## Sedang dikerjakan
- [ ] Stage A — `AspectStrategy` + registry, `describe_engines()`
- [ ] Stage B — proteksi koreksi ejaan, perluasan kosakata, ablasi A/B/C

## Berikutnya
- [ ] Stage C — gold set 500 baris, split 80/20, evaluasi rule-based/NB/SVM
- [ ] Stage D — `docs/`, notebook quickstart, demo chat dengan pilihan engine
- [ ] Demo B — LLM tool calling lewat `src/absa_library/tools.py` dan `demo/llm_tool.py`
- [ ] Stage E — CI, build wheel bersih, tag `v0.2.0`
- [ ] T8 — tuning kelas Netral pada NB/SVM (macro-F1 masih lemah)
- [ ] Fine-tuning IndoBERT nyata di GPU/Colab, simpan checkpoint
- [ ] `docs/` dan `notebooks/` (quickstart + perbandingan engine)
- [ ] Siapkan `experiments/` untuk tabel Bab 4

## Ide / belum diputuskan
- [ ] Ganti ke plugin hook kalau model sering lupa load skill `caveman`
- [ ] Tambah aspek `Aplikasi/Layanan/Harga` sebagai konfigurasi contoh di dokumentasi
