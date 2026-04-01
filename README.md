# Smart Parking System - Deployment Guide

## Quick Start (Without Docker)

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python3 run.py
```

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose installed

### Build and Run

1. Build the Docker image:
```bash
docker-compose build
```

2. Run the container (macOS/Linux):
```bash
# Allow X11 forwarding
xhost +local:docker

# Run the application
docker-compose up
```

3. Run the container (Windows):
```bash
# Install VcXsrv or Xming first
docker-compose up
```

## File Structure

```
smart-parking/
├── run.py                  # Entry point
├── import cv2.py           # Main application
├── line_detector.py        # Line detection utilities
├── yolo_detector.py        # YOLO detector
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── parking_slots.json      # Saved configurations
├── yolov8n.pt             # YOLO model (auto-downloaded)
├── data/                   # Sample data
├── images/                 # Reference images
└── videos/                 # Test videos
```

## Usage

1. **Load Video**: Click "Load Video" and select a parking lot video
2. **Capture Frame**: Click "Capture Frame" to grab a reference frame
3. **Draw Slots**: Click "Draw Slots Manually" to define parking areas
4. **Start Detection**: Select detection mode (Classic/YOLO) and click START

## Team

**Team**: High Stakes  
**Institution**: Rajagiri School of Engineering and Technology  
**Contact**: u2409008@rajagiri.edu.in

## License

See PROJECT_REPORT.md for full details.
