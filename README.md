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
| Konversi Braille ↔ Latin (Braille Indonesia tingkat 1) | selesai, teruji |
| Generator data sintetis | selesai |
| Model deteksi (BraillyNet) | terlatih (16 epoch sintetis + fine-tune foto asli) |
| Rekonstruksi baris/paragraf | selesai, teruji |
| Ekspor ONNX, CLI, HTTP API | selesai |
| Validasi pada Braille berbahasa Indonesia | **belum** (lihat [Keterbatasan](#keterbatasan)) |

## Instalasi

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Model siap pakai tersedia di `models/brailly.onnx` (12 MB, tanpa perlu PyTorch saat inferensi).

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

Beberapa pola dipakai ganda — tanda kutip buka dan tanda tanya memakai titik 236, kurung buka
dan tutup memakai titik 2356 — sehingga dibedakan dari posisinya dalam kata. Tabelnya dapat
disesuaikan lewat `BrailleProfile` bila buku sumber Anda memakai konvensi berbeda:

```python
from brailly.braille import BrailleCodec, BrailleProfile
profil = BrailleProfile(punctuation={",": "2", ".": "256", "?": "236"})
codec = BrailleCodec(profil)
```

## Keterbatasan

1. **Belum diuji pada Braille berbahasa Indonesia.** Foto uji berisi teks Rusia, jadi angka di
   atas mengukur ketepatan membaca *titik*, bukan hasil terjemahan bahasa Indonesia. Kode titik
   Braille bersifat universal, tetapi tabel tanda bacanya perlu diverifikasi.
2. **Tabel tanda baca perlu diverifikasi** terhadap dokumen resmi Sistem Simbol Braille
   Indonesia (khususnya kutip, kurung, titik dua, dan pemisah desimal/ribuan). Tabel saat ini
   disusun dari konvensi umum, bukan dari dokumen resmi.
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
  serve.py          API HTTP
models/brailly.onnx model terlatih siap pakai
tests/              uji konversi, tata letak, dan alur target->teks
```

## Rujukan

- Ovodov, I. G. "Optical Braille Recognition Using Object Detection Neural Network", ICCV 2021 —
  rumusan tugas dan metrik evaluasi.
- [Angelina Braille Images Dataset](https://github.com/IlyaOvodov/AngelinaDataset) — foto
  berlabel untuk fine-tune dan evaluasi (format anotasi CSV mengikuti dataset ini).
