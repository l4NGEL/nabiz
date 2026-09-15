# Faz 5 — protokol demosu

- üretilen tespit: 12 (2 anomali)
- MQTT broker [localhost:1883]: ULAŞILAMADI (beklenen — Docker/mosquitto çalışmıyorsa)
- doğrudan yayınlanan: 0  ·  yerel tampona yazılan: 12
- toplayıcı [http://localhost:8700]: 12 kayıt alındı

Canlı doğrulama için:  `docker compose up`  (mosquitto + toplayıcı), sonra bu betiği tekrar çalıştır — RB-03 (tampon/backfill) davranışını görmek için önce broker'sız çalıştır (mesajlar tamponlanır), sonra broker'ı başlat ve `TespitYayincisi`'nin backfill'ini gözlemle.