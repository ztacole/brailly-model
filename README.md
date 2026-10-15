# Brailly — model pembaca Braille berbasis kamera

Mengubah **foto halaman Braille** (huruf timbul) menjadi **teks Latin yang tersusun dalam paragraf**,
untuk aplikasi yang membantu orang awam membaca karya tulis Braille.

Alur: foto → deteksi sel Braille → susun baris & spasi → kelompokkan paragraf → terjemahkan ke huruf Latin.

```
foto kamera ──► BraillyNet (deteksi sel + 6 titik) ──► tata letak ──► kode Braille ──► teks Latin
```

## Status

| Komponen | Status |
|---|---|
| Konversi Braille ↔ Latin (Braille Indonesia tingkat 1) | selesai, 10 test lulus |
| Generator data sintetis | selesai |
| Model deteksi (BraillyNet) | terlatih (16 epoch sintetis + fine-tune foto asli) |
| Rekonstruksi baris/paragraf | selesai, teruji |
| Ekspor ONNX & TFLite, CLI, HTTP API | selesai |
| Validasi pada Braille berbahasa Indonesia | **belum** (lihat [Keterbatasan](#keterbatasan)) |

## Instalasi

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Model siap pakai: `models/brailly.onnx` (server/iOS/web) dan `models/brailly.tflite`
(Android, 3,4 MB). Inferensi tidak memerlukan PyTorch.

## Pemakaian

```bash
# satu foto -> teks
python scripts/infer.py foto.jpg --model models/brailly.onnx

# tampilkan juga kode Braille-nya + simpan visualisasi kotak deteksi
python scripts/infer.py foto.jpg --model models/brailly.onnx --show-braille --vis hasil.jpg

# foto miring/terbalik: potong & luruskan kertas, coba juga arah 180 derajat
python scripts/infer.py foto.jpg --model models/brailly.onnx --detect-page --rotate-180
```

Sebagai API HTTP (untuk aplikasi mobile/web):

```bash
BRAILLY_MODEL=models/brailly.onnx uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
curl -F "file=@foto.jpg" "http://localhost:8000/read"
```

Keluarannya JSON: `text` (seluruh teks), `paragraphs` (tiap paragraf, dengan `kind`
`paragraph`/`heading`), `lines` (teks + kode Braille Unicode per baris), dan `meta`
(jumlah sel, sudut kemiringan, jarak antar sel).

Dari Python:

```python
import cv2
from brailly.pipeline import BraillyReader, ReaderConfig

reader = BraillyReader("models/brailly.onnx", ReaderConfig(detect_page=True))
hasil = reader.read(cv2.imread("foto.jpg"))
print(hasil["text"])
```

## Cara kerja

**Model** (`brailly/model.py`) — MobileNetV3-Large (pra-latih ImageNet) dengan jalur
atas-bawah seperti FPN, keluaran pada 1/4 resolusi masukan. Pendekatannya bebas-anchor
(gaya CenterNet): satu peta titik-pusat sel, enam keluaran biner untuk titik 1–6, serta
regresi geseran sub-piksel dan ukuran sel. Ukuran ONNX ±11 MB.

Enam titik diprediksi **terpisah**, bukan sebagai 63 kelas. Dengan begitu pola yang jarang
muncul (tanda baca, tanda angka) tetap terbaca karena model hanya perlu mengenali titik,
bukan menghafal tiap kombinasi.

**Data sintetis** (`brailly/synth.py`) — halaman Braille dibuat secara prosedural lalu
dirender seperti foto ponsel: titik timbul sebagai peta ketinggian dengan arah cahaya acak
(kilau + bayangan), titik *interpoint* dari sisi balik kertas sebagai cekungan yang **tidak**
boleh terdeteksi, teks cetak di antara baris, tekstur kertas, kotoran, tepi halaman, latar
kayu/kain/berbintik, rotasi dan perspektif, blur, derau, serta artefak JPEG.

Tekstur latar yang menyerupai titik sengaja diperbanyak: pada evaluasi awal, tekstur kayu
dan kain menghasilkan ribuan deteksi palsu.

**Tata letak** (`brailly/layout.py`) — sudut baris diperkirakan dari tetangga horizontal
terdekat, sel dirunut menjadi baris, jarak antar sel dipakai untuk menyimpulkan spasi antar
kata, lalu baris dikelompokkan menjadi paragraf dari indentasi, jarak baris, dan baris yang
berakhir lebih awal. Baris yang terpusat dan pendek ditandai sebagai judul.

**Penyaringan ukuran** (`brailly/pipeline.py`) — deteksi berjalan dua kali: hasil pertama
dipakai untuk mengukur tinggi sel sebenarnya, gambar diskalakan agar sel ±36 px, lalu
dideteksi ulang. Deteksi yang ukurannya jauh dari modus (berbobot keyakinan) dibuang —
inilah penyaring utama terhadap deteksi palsu dari tekstur latar.

## Hasil

Diuji pada [Angelina Braille Images Dataset](https://github.com/IlyaOvodov/AngelinaDataset)
(foto buku Braille dan tulisan tangan siswa). `sym F1` = sel ditemukan **dan** keenam
titiknya benar — metrik yang dipakai pada makalah Optical Braille Recognition.
`braille CER` = jarak edit pada seluruh halaman dalam urutan baca, termasuk spasi.

Set validasi (43 foto buku + 6 tulisan tangan):

| Model | det F1 | sym F1 | braille CER |
|---|---|---|---|
| Sintetis saja (tanpa satu pun foto asli) | 0.880 | 0.775 | 0.415 |
| Setelah fine-tune foto asli | **0.997** | **0.987** | **0.017** |

Set uji `uploaded` — 50 foto acak dari pengguna layanan Braille, **tidak dipakai saat latih**.
Isinya foto apa adanya: gelap, miring, terpotong, berlatar ramai:

| Model | det F1 | sym F1 | braille CER |
|---|---|---|---|
| Sintetis saja | 0.939 | 0.879 | 0.179 |
| Setelah fine-tune foto asli | **0.968** | **0.938** | **0.088** |

Kecepatan ±0,4 detik per foto (CPU 5 utas, sisi panjang 1024 px).

Angka "sintetis saja" penting untuk kasus Anda: model yang **belum pernah melihat satu pun
foto asli** sudah mencapai sym F1 0.88. Artinya, bila nanti Anda punya foto buku Braille
Indonesia, generator sintetis bisa disetel ke format tersebut tanpa perlu banyak data berlabel.

### Uji ujung-ke-ujung bahasa Indonesia

Karena foto berlabel yang tersedia berbahasa Rusia, rantai lengkap diuji dengan halaman
Braille berisi teks Indonesia yang sudah diketahui, dirender menyerupai foto (pencahayaan
timbul, kemiringan, derau, kompresi JPEG):

```bash
python scripts/demo_roundtrip.py --model models/brailly.onnx
```

Hasilnya 10/10 baris tepat, CER 0.0000, dan keempat paragraf terkelompok benar — termasuk
angka (`17`, `1945`), tanda tanya, titik, dan koma.

## ONNX atau TFLite untuk Android?

Keduanya sudah diekspor dan **diukur pada 50 foto uji yang sama** lewat jalur
pasca-pemrosesan yang sama, jadi perbedaannya murni dari model. Kolom "APK"
adalah ukuran terkompresi: model + runtime arm64 (terukur dari AAR-nya —
ONNX Runtime 12,9 MB, LiteRT 1,7 MB).

| varian | APK | det F1 | sym F1 | CER | detik* |
|---|---|---|---|---|---|
| onnx-fp32 | 24,1 MB | 0.968 | **0.938** | **0.088** | 0,15 |
| onnx-int8 | 15,2 MB | 0.965 | 0.935 | 0.092 | 0,45 |
| tflite-fp32 | **13,0 MB** | 0.968 | 0.931 | 0.098 | 0,30 |
| tflite-dyn-int8 | **4,5 MB** | 0.966 | 0.922 | 0.107 | 0,34 |
| tflite-static-int16 | 4,6 MB | 0.959 | 0.921 | 0.110 | 5,4 |
| tflite-static-int8 | 4,6 MB | 0.961 | 0.907 | 0.122 | 0,24 |

\*Waktu diukur di CPU x86, **bukan** penunjuk kecepatan di ARM. int16 lambat di
CPU karena dirancang untuk NPU.

Dua temuan yang menentukan:

**1. TFLite tidak kalah akurat dari ONNX.** Selisih fp32 (0.938 → 0.931) ternyata
sama sekali bukan karena TFLite, melainkan karena **bentuk masukan tetap**. Saat
ONNX dipaksa melewati jalur letterbox yang sama, hasilnya **identik**: sym F1
0.9313, CER 0.0984 — angka yang sama persis dengan tflite-fp32. Jadi biaya 0,7
poin itu adalah harga bentuk statis, yang justru dibutuhkan ponsel untuk
delegate GPU/NPU.

**2. Yang mahal adalah kuantisasi aktivasi, bukan bobot.** Mode `static-int8`
turun 3,1 poin karena logit heatmap berjangkauan ±15 terpotong oleh skala int8.
Mode `dynamic-int8` (bobot int8, aktivasi fp32) hanya turun 0,9 poin dengan
ukuran APK nyaris sama — 4,5 MB.

**Rekomendasi: `tflite-dyn-int8`.** Dibanding onnx-fp32, APK-nya **5× lebih
kecil** (4,5 vs 24,1 MB) dengan biaya 1,6 poin sym F1. Untuk aplikasi
aksesibilitas yang menyasar ponsel kelas bawah, pertukaran itu sepadan.
Bila akurasi jadi prioritas utama dan 13 MB masih dapat diterima, `tflite-fp32`
praktis setara ONNX.

```bash
python scripts/export_tflite.py --ckpt runs/ft/best.pt \
    --out models/brailly.tflite --quant-mode dynamic-int8

# bandingkan sendiri varian apa pun
python scripts/bench_backends.py --list data/uploaded/test.txt \
    --model onnx=models/brailly.onnx --model tflite=models/brailly.tflite
```

ONNX tetap dipertahankan karena dipakai `serve.py` (jalur server) dan berguna
untuk iOS/web. `BraillyReader` menerima `.onnx`, `.tflite`, maupun `.pt` —
backend-nya dipilih otomatis dari ekstensi berkas.

Catatan: ekspor TFLite memerlukan `torch<2.14` sehingga bentrok dengan
lingkungan latih. Pakai virtualenv terpisah:

```bash
python -m venv .venv-tflite && . .venv-tflite/bin/activate
pip install "torch==2.13.*" litert-torch opencv-python-headless
```

## Melatih ulang

```bash
# 1. pra-latih dengan data sintetis (tanpa perlu data berlabel)
python scripts/train.py --out runs/synth --epochs 16 --synth-per-epoch 3200

# 2. fine-tune dengan foto asli berlabel (format CSV Angelina)
python scripts/train.py --out runs/ft --init runs/synth/best.pt \
    --real-list data/books/train.txt data/handwritten/train.txt \
    --real-ratio 0.5 --epochs 6 --lr 5e-4

# 3. evaluasi & ekspor
python scripts/evaluate.py --model runs/ft/best.pt --list data/books/val.txt
python scripts/export_onnx.py --ckpt runs/ft/best.pt --out models/brailly.onnx
```

`--resume` melanjutkan dari `<out>/last.pt` (termasuk status optimizer), dan
`--epochs-per-run N` berhenti setelah N epoch — berguna bila waktu proses terbatas.

Anotasi memakai format CSV Angelina: `kiri;atas;kanan;bawah;label` per sel, koordinat
dinormalisasi 0–1, `label` = bitmask titik 1–63 (bit 0 = titik 1). Gambar **tanpa** file CSV
diperlakukan sebagai contoh negatif (foto bukan Braille), berguna untuk menekan deteksi palsu.

## Tabel Braille

`brailly/braille.py` memuat profil Braille Indonesia tingkat 1 (tanpa singkatan): huruf a–z,
tanda kapital (titik 6; digandakan untuk kata berhuruf kapital semua), tanda angka (titik 3456),
tanda huruf (titik 56), dan tanda baca.

Tabel tanda bacanya sudah diverifikasi cocok dengan **standar Braille internasional**
(lihat [artikel Braille, bagian Punctuation](https://en.wikipedia.org/wiki/Braille)), dan
dikunci oleh `tests/test_braille.py`:

| Tanda | Titik | Tanda | Titik |
|---|---|---|---|
| koma `,` | 2 | tanda seru `!` | 235 |
| titik koma `;` | 23 | tanda tanya `?` | 236 |
| titik dua `:` | 25 | kutip buka `"` | 236 |
| titik `.` | 256 | kutip tutup `"` | 356 |
| apostrof `'` | 3 | kurung `(` `)` | 2356 |
| tanda hubung `-` | 36 | garis miring `/` | 34 |

Beberapa pola dipakai ganda — tanda kutip buka dan tanda tanya sama-sama titik 236, kurung
buka dan tutup sama-sama titik 2356 — sehingga dibedakan dari posisinya dalam kata, sesuai
aturan standar tersebut. Tabelnya dapat disesuaikan lewat `BrailleProfile` bila buku sumber
Anda memakai konvensi berbeda:

```python
from brailly.braille import BrailleCodec, BrailleProfile
profil = BrailleProfile(punctuation={",": "2", ".": "256", "?": "236"})
codec = BrailleCodec(profil)
```

## Keterbatasan

1. **Belum diuji pada Braille berbahasa Indonesia.** Foto uji berisi teks Rusia, jadi angka di
   atas mengukur ketepatan membaca *titik*, bukan hasil terjemahan bahasa Indonesia. Kode titik
   Braille bersifat universal, tetapi tabel tanda bacanya perlu diverifikasi.
2. **Tabel tanda baca** sudah dicocokkan dengan standar Braille internasional dan dikunci
   oleh test, tetapi **belum** dibandingkan dengan dokumen resmi Sistem Simbol Braille
   Indonesia (dokumen tersebut tidak tersedia daring saat pengerjaan). Bila ternyata ada
   simbol khas Indonesia yang berbeda, sesuaikan lewat `BrailleProfile` — tidak perlu
   melatih ulang model, karena model hanya mengenali titik, bukan maknanya.
3. Hanya mendukung **Braille tingkat 1 (tanpa singkatan)** dan sel 6 titik.
4. Halaman diasumsikan tegak; gunakan `--rotate-180` untuk halaman terbalik. Rotasi 90° belum didukung.
5. Dataset Angelina **tidak mencantumkan lisensi**. Periksa izin pemakaiannya sebelum dipakai
   untuk produk komersial. Bagian sintetis sepenuhnya bebas dari masalah ini.

## Struktur

```
brailly/
  braille.py      konversi kode Braille <-> teks Latin
  synth.py        generator halaman Braille sintetis
  textgen.py      kalimat acak bahasa Indonesia
  model.py        arsitektur BraillyNet
  data.py         dataset sintetis & foto asli, penyandian target latih
  losses.py       focal loss untuk peta pusat + BCE titik + L1 regresi
  postprocess.py  keluaran jaringan -> daftar sel
  layout.py       sel -> baris -> paragraf
  preprocess.py   deteksi & perataan lembar kertas
  pipeline.py     BraillyReader (alur lengkap, backend PyTorch/ONNX)
  metrics.py      pencocokan sel & jarak edit
scripts/
  train.py          latih (sintetis dan/atau foto asli), dapat dilanjutkan
  evaluate.py       hitung det/sym F1 dan CER
  infer.py          baca satu foto dari CLI
  demo_roundtrip.py uji ujung-ke-ujung dengan teks Indonesia
  export_onnx.py    ekspor ke ONNX + verifikasi numerik
  export_tflite.py  ekspor ke TFLite/LiteRT (4 mode kuantisasi) + verifikasi
  bench_backends.py bandingkan varian model: akurasi, ukuran, kecepatan
  serve.py          API HTTP
models/
  brailly.onnx      model terlatih (server, iOS, web)
  brailly.tflite    model terlatih (Android, dynamic-int8)
tests/              uji konversi, tata letak, dan alur target->teks
```

## Rujukan

- Ovodov, I. G. "Optical Braille Recognition Using Object Detection Neural Network", ICCV 2021 —
  rumusan tugas dan metrik evaluasi.
- [Angelina Braille Images Dataset](https://github.com/IlyaOvodov/AngelinaDataset) — foto
  berlabel untuk fine-tune dan evaluasi (format anotasi CSV mengikuti dataset ini).
