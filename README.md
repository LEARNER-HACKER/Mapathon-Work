# 🚗 Mapathon-Work  
AI-powered smart parking detection system using YOLOv8 and OpenCV to identify real-time parking slot occupancy from images and video.

---

# 🚗 AI-Based Smart Parking Detection System

## 📌 Overview
This project is a computer vision-based smart parking system developed during a hackathon. It detects vehicle occupancy in parking lots using YOLOv8 and OpenCV.

## 🎯 Problem Statement
Urban areas face parking congestion due to lack of real-time parking visibility.

## 💡 Solution
Our system:
- Detects vehicles using YOLOv8
- Identifies parking slot boundaries
- Classifies slots as **Occupied** or **Available**
- Stores results in structured format (CSV/JSON)

---

## 🛠 Tech Stack
- Python
- OpenCV
- YOLOv8
- Docker

---

## 📂 Project Structure

```
mapathon-work/
│
├── app/
│   ├── run.py
│   ├── yolo_detector.py
│   ├── line_detector.py
│
├── data/
│   ├── parking_slots.json
│   ├── sample_images/
│   ├── sample_videos/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/mapathon-work.git
cd mapathon-work
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### 3️⃣ Download YOLOv8 Model

Install Ultralytics:

```bash
pip install ultralytics
```

Then run once in Python to automatically download the model:

```python
from ultralytics import YOLO
YOLO("yolov8n.pt")
```

OR manually download `yolov8n.pt` and place it in the project root directory.

---

### 4️⃣ Verify Parking Slot Configuration

Ensure the following file exists and contains parking slot coordinates:

```
parking_slots.json
```

---

### 5️⃣ Run the Application

If `run.py` is in the root folder:

```bash
python run.py
```

If it is inside an `app` folder:

```bash
python app/run.py
```

---

## ✅ Expected Output

- The system processes image/video input  
- Parking slots are marked visually  
- Occupied and Available slots are displayed  
- Parking status is stored in CSV format  

---

## 🐳 (Optional) Run Using Docker

If Docker is installed:

```bash
docker-compose up --build
```

---

## 🚀 Future Improvements
- Real-time web dashboard
- IoT sensor integration
- Smart city deployment support

---

## 👥 Team
(Add team member names here)

---
