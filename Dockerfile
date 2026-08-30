# syntax=docker/dockerfile:1

# ---------- base ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl tini \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------- deps ----------
FROM base AS deps
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- runtime (alpr-api) ----------
FROM deps AS runtime
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps -e .
# Model weights are mounted at runtime under /app/models (see docker-compose.yml).
ENV TZ_ALPR_API_HOST=0.0.0.0 \
    TZ_ALPR_API_PORT=8080 \
    TZ_ALPR_ENV=production \
    TZ_ALPR_DEVICE=cpu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "tz_alpr.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# ---------- training (optional, GPU base recommended in practice) ----------
FROM deps AS training
RUN pip install torch torchvision lightning albumentations pandas scikit-learn tqdm ultralytics \
      onnx onnxruntime tensorboard
COPY . .
RUN pip install --no-deps -e .
CMD ["python", "training/train_ocr.py", "--help"]
