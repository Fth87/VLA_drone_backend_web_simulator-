# Drone VLA Backend 
Backend ini dipakai untuk inferensi Drone VLA dan menjadi sumber aksi untuk frontend.

## Tujuan
- Menjalankan model Drone VLA dari notebook Kaggle.
- Membuka endpoint publik untuk diakses frontend.

## Cara Menjalankan

1. Buka file notebook berikut di Kaggle:
   - [backend.ipynb](backend.ipynb)
2. Pastikan GPU aktif.
3. Klik Run All.
4. Tunggu sampai backend publik aktif dan URL endpoint muncul.
5. Salin URL publik tersebut.
6. Tempel URL ke environment frontend sebagai base URL backend.
7. Jalankan frontend.

## Integrasi Frontend

Repo frontend:

- https://github.com/Fth87/VLA_drone_web_simulator

Frontend cukup butuh satu nilai environment:

- VITE_VLA_API_URL
  - Isi dengan URL publik dari output notebook Kaggle.

Contoh nilai:

https://xxxxx.gradio.live

Sesudah itu frontend dapat memanggil endpoint infer dan health pada backend yang sama.

## Kontrak API 

Endpoint utama:

- POST /infer
- GET atau POST /health

Payload infer menggunakan multipart form data dengan field:

- image_input (wajib)
- task_input (wajib)
- state_input (opsional, format list string)

Response infer:

- success
- first_action
- trajectory
- inference_time_ms
- error

## Ringkasan 

- Notebook menyiapkan dependency dan model checkpoint.
- Runtime patch memastikan kompatibilitas OpenPI dan loader model.
- Policy dibuat sekali lalu dipakai untuk inferensi berulang.
- Backend publik diexpose agar frontend bisa memanggil inferensi real time.

