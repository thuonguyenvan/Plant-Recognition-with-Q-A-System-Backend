FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY main.py .
COPY cv_class_to_vietnamese.json .
COPY class.txt .

# Copy service modules
COPY api/ ./api/
COPY services/ ./services/
COPY utils/ ./utils/
COPY scripts/ ./scripts/

# Copy data directories (essential for the app)
COPY data/ ./data/
COPY inat_representative_photos/ ./inat_representative_photos/

# Expose Hugging Face Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
