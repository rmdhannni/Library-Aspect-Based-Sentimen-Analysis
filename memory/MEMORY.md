# Memory

## 2026-11-25 — Perencanaan 80% sidang progress, demo, dan push GitHub

Tujuan: menetapkan target 80% untuk sidang progress, memvalidasi cakupan MVP,
menguji demo satu file, dan menyiapkan repository GitHub.

Keputusan:
- Target 80% ditempatkan pada pertengahan atau akhir Januari 2027. Definisi 80%
  adalah 80% ruang lingkup MVP ditambah Bab 4 awal. Sisa 20% mencakup
  fine-tuning IndoBERT nyata, perbandingan empat engine penuh, rilis PyPI, dan
  penulisan bab.
- MVP wajib tetap berjalan dengan satu engine saja, yaitu `rule-based`, tanpa
  model pretrained dan tanpa internet. Engine NB, SVM, dan IndoBERT bersifat
  opsional serta harus gagal dengan pesan informatif, bukan crash.
- Evaluasi memakai dua lapis: gold set open-domain untuk membuktikan klaim
  general, dan subset aspek `Aplikasi/Layanan/Harga` untuk tabel perbandingan.
- Gold set dinaikkan menjadi 500 baris dengan komposisi usulan 250 ulasan AI
  Chatbots, 125 e-commerce, dan 125 edukasi atau film. Anotasi memakai
  pre-label otomatis, verifikasi penuh oleh penulis, dan 100 baris sampel oleh
  anotator kedua untuk Cohen's kappa.
- Engine untuk sidang progress: rule-based, Naive Bayes, dan SVM. IndoBERT
  tetap kerangka sampai setelah gate 80%.
- Koreksi ejaan akan memakai proteksi kata dan diputuskan lewat ablasi tiga
  konfigurasi: jarak 2 tanpa proteksi, jarak 1, serta jarak 2 dengan proteksi
  dan kosakata umum. Default boleh berubah menjadi nonaktif atau jarak 1 bila
  data mendukung, dan perubahan dari PRD dicatat sebagai deviasi.
- Demo A wajib: aplikasi chat sederhana dengan pilihan engine. Demo B menjadi
  nilai tambah: LLM memanggil library lewat tool calling. LLM hanya menjadi
  orkestrator, label tetap berasal dari library. Rencana teknis Demo B adalah
  `src/absa_library/tools.py`, `demo/llm_tool.py`, dan
  `docs/integrations/llm.md`.
- Dokumentasi dan notebook memakai Bahasa Indonesia. Docstring dan nama API
  tetap Bahasa Inggris.
- Nama paket `absa-library`, nama import `absa_library`, lisensi MIT.
- Repo GitHub yang dipakai:
  https://github.com/rmdhannni/Library-Aspect-Based-Sentimen-Analysis.git
  Repo masih kosong dan belum memiliki commit.
- Bimbingan 26 November 2026 akan memvalidasi empat hal: cukup tiga engine atau
  wajib empat, bolehkah window-based fallback menggantikan POS/dependency,
  bolehkah pre-label otomatis dengan verifikasi manusia, dan bentuk demo yang
  diterima.

Temuan penting:
- Ekstraksi aspek general secara desain, tetapi belum general secara kualitas.
  Contoh kegagalan: `akting` berubah menjadi `paking`, `materi` menjadi
  `baterai`, dan `dosen` menjadi `pesen` karena koreksi ejaan terlalu agresif.
- Leksikon opini hanya 156 kata positif dan 109 kata negatif. Kata `habis`,
  `kaku`, dan `padat` belum tersedia.
- Uji `kameranya bagus tapi memorynya cepat habis`: rule-based menghasilkan
  kamera Positif dan memory Positif (salah), Naive Bayes menghasilkan Netral
  dan Positif (salah), SVM menghasilkan Positif dan Negatif (benar tetapi
  confidence hanya sekitar 0,39). Penyebabnya model dilatih level kalimat,
  bias token `memorynya` dan `kamera-nya`, leksikon `habis` yang kosong, dan
  akhiran `-nya` yang belum dinormalisasi.
- Kondisi library saat ini: 294 test lulus, coverage 97,22%, ruff bersih,
  wheel 1,99 MB, build dan twine check lulus. Metrik pretrained level kalimat:
  Naive Bayes akurasi 64,41% dan macro-F1 53,57%; SVM akurasi 66,99% dan
  macro-F1 49,52% pada 4.707 data uji.
- Repo lokal belum memiliki commit, branch masih `master`, dan remote GitHub
  belum dipasang. `Data/`, `dist/`, dan `.venv/` sudah di-ignore. Model
  pretrained tetap ikut karena ada aturan unignore.
- `demo_absa.py` sudah dibuat pengguna dan berjalan. Mode interaktif memilih
  engine 1/2/3, sedangkan `--demo` menjalankan empat kalimat dengan rule-based
  dan SVM.

File yang disentuh: `demo_absa.py` dibuat oleh pengguna. Tidak ada perubahan
kode library pada sesi ini.

Blocker: keputusan pembimbing, kredensial GitHub, dan pelabelan gold set masih
menunggu.

Langkah lanjut: eksekusi Stage A (kontrak strategy aspek, registry,
`describe_engines()`, commit awal, push GitHub), lanjut Stage B (proteksi
koreksi ejaan dan ablasi), kemudian Stage C (gold set 500 baris dan evaluasi
tiga engine).

## 2026-11-21 — Implementasi library ABSA T1–T7

Tujuan: membangun paket Python `absa-library` sesuai `PRD.md` untuk milestone
T1 sampai T7 (library fungsional + test), setelah membersihkan folder `Data/`.

Keputusan:
- Aspek default bersifat open-domain (`aspects="auto"`). Mode kamus dan
  `aspects=None` tetap tersedia; `Aplikasi/Layanan/Harga` hanya contoh konfigurasi.
- Tiga kelas sentimen: `Positif`, `Negatif`, `Netral`. Rule-based memakai
  `neutral_margin=0.3` dan distribusi probabilitas berbasis sigmoid tergeser.
- NB memakai `ComplementNB(alpha=0.3)`; SVM memakai `LinearSVC` yang dibungkus
  `CalibratedClassifierCV(cv=3, method="sigmoid")`.
- Model pretrained NB dan SVM dilatih pada dataset AI Chatbots (23.532 ulasan
  unik) dan dibundel di `src/absa_library/data/pretrained/`.
- IndoBERT belum di-fine-tune nyata. Kerangka train/predict/save/load sudah ada
  dan diuji dengan backend palsu, sesuai kesepakatan.
- Kosakata koreksi ejaan diturunkan dari frekuensi token PRDECT-ID (CC BY 4.0).
  Leksikon sentimen, negasi, intensifier, dan kosakata aspek ditulis sendiri (MIT).
- Hasil build diverifikasi: wheel 1,99 MB, `twine check` lulus, instalasi di venv
  bersih berhasil, dan `import absa_library` tidak memuat `torch`/`transformers`.

File yang disentuh (utama): `pyproject.toml`, `README.md`, `CHANGELOG.md`,
`NOTICE`, `.gitignore`, `src/absa_library/*.py`,
`src/absa_library/engines/*.py`, `src/absa_library/data/*`,
`tests/*.py`, `experiments/build_package_data.py`,
`experiments/train_pretrained.py`.

Hasil verifikasi: 294 test lulus, coverage 97,22%, `ruff check` bersih,
`python -m build` menghasilkan `.whl` dan `.tar.gz`.

Blocker: belum ada. Catatan: akurasi pretrained NB 64,4% dan SVM 67,0% dengan
macro-F1 0,54 dan 0,50; kelas Netral masih lemah dan perlu dituning pada T8.

Langkah lanjut: putuskan commit awal, lalu masuk T8 (dataset berlabel, evaluasi
4 engine, tuning Netral, fine-tuning IndoBERT di Colab).

## 2026-11-21 — Setup caveman + memori

Tujuan: buat setiap prompt di `C:\laragon\www\TA` memakai skill caveman, dan simpan memori kerja.

Keputusan:
- Aktivasi caveman lewat `AGENTS.md` di root proyek (opsi A).
- Matikan sementara dengan "stop caveman" / "normal mode".
- Semua memori sesi dalam satu file `memory/MEMORY.md`, di-track git.
- Catatan memori ditulis prosa normal, bukan gaya caveman.

File disentuh: `AGENTS.md` (dibuat, lalu ditambah bagian `## Memory`), `TODO.md` (dibuat),
`memory/MEMORY.md` (dibuat).

Blocker: belum ada.

Langkah lanjut: putuskan commit awal, lalu mulai dokumentasi proyek TA.
