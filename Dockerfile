FROM python:3.11-slim

WORKDIR /app

# system deps for faiss-cpu + sqlite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# faiss_index dir must be baked in or mounted at runtime
# data/ dir is created by init_db() on startup
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]