"""Random Indonesian-like text used to drive the synthetic page generator."""

from __future__ import annotations

import random

WORDS = """
yang dan di ini itu dengan untuk tidak dari dalam akan pada juga saya ke karena
tersebut bisa ada mereka lebih kami sudah atau adalah oleh kita dia telah harus
kepada seperti hanya banyak saat bahwa semua tahun orang baru anak hari waktu
rumah sekolah buku membaca menulis belajar guru murid kota desa jalan pagi malam
siang sore besar kecil baik indah cantik senang sedih cepat lambat tinggi rendah
air api tanah udara laut gunung sungai hutan pohon bunga daun buah burung ikan
kucing anjing kuda sapi ayam makan minum tidur bangun pergi datang pulang lihat
dengar bicara tanya jawab cerita dongeng raja ratu putri pangeran negeri rakyat
ibu ayah kakak adik nenek kakek keluarga teman sahabat cinta kasih sayang hidup
mata tangan kaki kepala hati jiwa pikiran ilmu pengetahuan bahasa indonesia
braille titik huruf kata kalimat paragraf halaman bab judul pelajaran latihan
pemerintah masyarakat pendidikan kesehatan ekonomi budaya sejarah alam dunia
matahari bulan bintang langit hujan angin awan pelangi musim kemarau panen padi
sawah ladang petani nelayan pedagang pasar uang harga barang toko kantor kerja
perjalanan kendaraan kereta kapal pesawat sepeda mobil motor jembatan pelabuhan
merdeka bangsa negara pancasila garuda nusantara pulau jawa sumatra kalimantan
sulawesi papua bali jakarta bandung surabaya yogyakarta medan makassar semarang
tetapi namun kemudian lalu setelah sebelum ketika jika maka agar supaya walaupun
sangat paling sekali selalu sering kadang jarang pernah belum masih sedang mulai
pertama kedua ketiga terakhir setiap beberapa seluruh sebagian para tentang
kemerdekaan perpustakaan penelitian pengalaman perkembangan kebersamaan
""".split()

NAMES = ["Andi", "Budi", "Siti", "Dewi", "Rina", "Agus", "Joko", "Putri", "Sari",
         "Tono", "Indonesia", "Jakarta", "Bandung", "Allah", "Tuhan", "Senin", "Minggu"]


def random_word(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.06:
        return str(rng.choice([rng.randint(0, 9), rng.randint(10, 99), rng.randint(100, 2030),
                               rng.randint(1000, 999999)]))
    if r < 0.12:
        return rng.choice(NAMES)
    if r < 0.14:
        w = rng.choice(WORDS)
        return f"{w}-{w}"  # reduplication (anak-anak)
    if r < 0.15:
        return rng.choice(WORDS).upper()
    return rng.choice(WORDS)


def random_sentence(rng: random.Random) -> str:
    n = rng.randint(3, 14)
    words = [random_word(rng) for _ in range(n)]
    words[0] = words[0][:1].upper() + words[0][1:]
    for i in range(1, n - 1):
        if rng.random() < 0.08:
            words[i] += rng.choice([",", ",", ";", ":"])
    if rng.random() < 0.06:
        a = rng.randrange(n)
        words[a] = '"' + words[a]
        b = rng.randrange(a, n)
        words[b] = words[b] + '"'
    if rng.random() < 0.04:
        a = rng.randrange(n)
        words[a] = "(" + words[a] + ")"
    words[-1] += rng.choice([".", ".", ".", ".", "?", "!"])
    return " ".join(words)


def random_paragraph(rng: random.Random) -> str:
    return " ".join(random_sentence(rng) for _ in range(rng.randint(1, 5)))
