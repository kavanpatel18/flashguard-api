FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api_server.py .
COPY improved_minute_model.keras .
COPY improved_flash_crash_model.keras .
COPY frontend/ ./frontend/

# Hugging Face Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "api_server:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
