# Wan 2.2 TI2V-5B – RunPod Serverless

Pipeline serverless pour générer de la **vidéo 720p @ 24 fps** (T2V/I2V) avec Wan 2.2 via Diffusers.

## Build & Push (Docker Hub)
```bash
docker build -t <dockerhub_user>/wan22-serverless:latest .
docker push <dockerhub_user>/wan22-serverless:latest