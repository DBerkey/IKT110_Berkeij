# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends build-essential \ 
    && rm -rf /var/lib/apt/lists/*

# Copy the full repo so shared data/model files remain reachable via relative paths.
COPY . /app

RUN pip install --upgrade pip \
    && pip install -e ./dota/dota-oracle[prod]

ENV DORACLE_HOST=0.0.0.0 \
    DORACLE_PORT=8000 \
    WEB_CONCURRENCY=2

EXPOSE 8000

WORKDIR /app/dota/dota-oracle

CMD ["gunicorn", "-c", "gunicorn.conf.py", "doracle.app:app"]
