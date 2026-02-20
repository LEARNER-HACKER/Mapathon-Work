# Mapathon-Work
AI-powered smart parking detection system using YOLOv8 and OpenCV to identify real-time parking slot occupancy from images and video.
# 🚗 AI-Based Smart Parking Detection System

## 📌 Overview
This project is a computer vision-based smart parking system developed during a hackathon. It detects vehicle occupancy in parking lots using YOLOv8 and OpenCV.

## 🎯 Problem Statement
Urban areas face parking congestion due to lack of real-time parking visibility.

## 💡 Solution
Our system:
- Detects vehicles using YOLOv8
- Identifies parking slot boundaries
- Classifies slots as Occupied or Available
- Stores results in structured format (CSV/JSON)

## 🛠 Tech Stack
- Python
- OpenCV
- YOLOv8
- Docker

## 📂 Project Structure
Explain folders here

## ⚙️ Installation
▶️ How to Run the Project
3️⃣ Download YOLOv8 Model

Download the YOLOv8 Nano model:

pip install ultralytics

Then run once in Python (it auto-downloads the model):

from ultralytics import YOLO
YOLO("yolov8n.pt")

OR manually download yolov8n.pt and place it in the project root directory.

4️⃣ Verify Parking Slot Configuration

Ensure the file below exists and contains parking slot coordinates:

parking_slots.json
5️⃣ Run the Application

If run.py is in the root folder:

python run.py

If it is inside an app folder:

python app/run.py
6️⃣ Expected Output

The system processes image/video input

Parking slots are marked

Occupied and Available slots are displayed

Status is stored in CSV format

🐳 (Optional) Run Using Docker

If Docker is installed:

docker-compose up --build
1. Clone the repository
2. Install dependencies:
