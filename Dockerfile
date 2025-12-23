FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch (CPU version) to keep image size small.
# If you need GPU support, change the index-url to https://download.pytorch.org/whl/cu118 (or appropriate version)
# and use nvidia/cuda base image.
RUN pip install --no-cache-dir \
    "torch>=2.4.0" \
    "torchvision>=0.19.0" \
    "torchaudio>=2.4.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "scipy>=1.11.0" \
    "matplotlib>=3.8.0" \
    "scikit-learn>=1.3.0" \
    "tqdm>=4.66.0" \
    "networkx>=3.2.0" \
    "torch-geometric>=2.5.0" \
    "fastapi>=0.124.4" \
    "uvicorn>=0.38.0" \
    "python-multipart>=0.0.20"

# Copy source code and config
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY vocab.json .

# Copy checkpoints (required for inference)
COPY checkpoints_8/ ./checkpoints_8/

# Set python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
