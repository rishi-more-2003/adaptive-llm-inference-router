FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY benchmarks ./benchmarks
COPY runs/.gitkeep ./runs/.gitkeep

EXPOSE 8000

CMD ["python", "-m", "src.server", "--config", "configs/default.yaml"]
