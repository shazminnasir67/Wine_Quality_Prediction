# Wine Quality Prediction API - Docker Configuration

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ml mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Expose ports
EXPOSE 8000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "backend/main.py"]

# Alternative commands for different services:
# FastAPI only: docker run -p 8000:8000 wine-quality python backend/main.py  
# MLflow only: docker run -p 5000:5000 wine-quality mlflow ui --host 0.0.0.0
# Both services: docker run -p 8000:8000 -p 5000:5000 wine-quality