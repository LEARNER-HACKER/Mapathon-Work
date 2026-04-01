"""
YOLOv8 Vehicle Detector for Smart Parking System
Uses pretrained YOLOv8 model to detect vehicles in parking lots
"""

from ultralytics import YOLO
import cv2
import numpy as np


class YOLOVehicleDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.25):
        """
        Initialize YOLO vehicle detector
        
        Args:
            model_name: YOLO model to use (yolov8n.pt is fastest, good for real-time)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
                                 Lowered to 0.25 to better detect stationary parked vehicles
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        print(f"YOLO model loaded successfully! Confidence threshold: {confidence_threshold}")
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in the frame
        
        Args:
            frame: Input image/frame (BGR format)
            
        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_name: str
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Filter for vehicles only and confidence threshold
                if cls in self.vehicle_classes and conf >= self.confidence_threshold:
                    detections.append({
                        'bbox': box,
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': self.vehicle_classes[cls]
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input image
            detections: List of detections from detect_vehicles()
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            conf = det['confidence']
            cls_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return output
    
    def check_overlap(self, bbox, polygon):
        """
        Check if a bounding box overlaps with a parking slot polygon
        
        Args:
            bbox: [x1, y1, x2, y2] - vehicle bounding box
            polygon: np.array of polygon points defining parking slot
            
        Returns:
            bool: True if there's significant overlap
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Create mask for bounding box
        # Get bounding rect of polygon to determine mask size
        poly_x = polygon[:, 0]
        poly_y = polygon[:, 1]
        min_x, max_x = int(poly_x.min()), int(poly_x.max())
        min_y, max_y = int(poly_y.min()), int(poly_y.max())
        
        # Expand bounds to include both bbox and polygon
        final_min_x = min(min_x, x1)
        final_min_y = min(min_y, y1)
        final_max_x = max(max_x, x2)
        final_max_y = max(max_y, y2)
        
        width = final_max_x - final_min_x + 1
        height = final_max_y - final_min_y + 1
        
        # Create masks
        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        poly_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw bounding box on mask (adjusted coordinates)
        cv2.rectangle(bbox_mask,
                     (x1 - final_min_x, y1 - final_min_y),
                     (x2 - final_min_x, y2 - final_min_y),
                     255, -1)
        
        # Draw polygon on mask (adjusted coordinates)
        adjusted_polygon = polygon.copy()
        adjusted_polygon[:, 0] -= final_min_x
        adjusted_polygon[:, 1] -= final_min_y
        cv2.fillPoly(poly_mask, [adjusted_polygon.astype(np.int32)], 255)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(bbox_mask, poly_mask)
        intersection_area = cv2.countNonZero(intersection)
        
        # Calculate polygon area
        poly_area = cv2.countNonZero(poly_mask)
        
        if poly_area == 0:
            return False
        
        # Consider occupied if more than 15% of parking slot is covered
        # Lowered from 30% to better detect partially visible or stationary vehicles
        overlap_ratio = intersection_area / poly_area
        return overlap_ratio > 0.15
    
    def set_confidence_threshold(self, threshold):
        """Set new confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
