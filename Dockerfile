# Pinned to a specific Python minor version so results reproduce.
# Deliberately NOT 3.15: at the time of writing 3.12 is the version these
# scientific libraries are most widely tested against.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg

# libgomp1 is the OpenMP runtime that LightGBM links against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "import sklearn, shap; print('environment ready')"]
