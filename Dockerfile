FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Railway exposes only one port
EXPOSE 8501

# Start FastAPI in background, then Streamlit
CMD uvicorn backend.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.headless=true