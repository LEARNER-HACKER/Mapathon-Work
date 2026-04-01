FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run.py .
COPY "import cv2.py" .
COPY line_detector.py .
COPY yolo_detector.py .

# Copy YOLO model if exists
COPY yolov8n.pt . 2>/dev/null || true

# Create directories for data
RUN mkdir -p data images videos

# Set environment variables
ENV DISPLAY=:0

# Run the application
CMD ["python3", "run.py"]
