FROM python:3.10-slim
RUN apt-get update && apt-get install -y \
    gcc g++ git libgeos-dev libgdal-dev libspatialindex-dev curl make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data /app/outputs /app/logs
ENV PYTHONPATH=/app
EXPOSE 3000 8080 8000 8888
CMD ["python", "-m", "src.bcpc_pipeline"]
