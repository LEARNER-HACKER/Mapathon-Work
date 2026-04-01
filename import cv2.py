import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
from yolo_detector import YOLOVehicleDetector
from line_detector import LineBasedSlotDetector


class ParkingSlot:
    def __init__(self, points, slot_id):
        self.points = np.array(points, dtype=np.int32)
        self.slot_id = slot_id
        self.occupied = False
        
    def to_dict(self):
        return {
            'points': self.points.tolist(),
            'slot_id': self.slot_id
        }
    
    @staticmethod
    def from_dict(data):
        return ParkingSlot(data['points'], data['slot_id'])


class ParkingDetector:
    def __init__(self, use_yolo=False):
        self.slots = []
        self.use_yolo = use_yolo
        self.yolo_detector = None
        
        if use_yolo:
            try:
                print("Initializing YOLO detector...")
                # Lower confidence threshold to better detect stationary parked vehicles
                self.yolo_detector = YOLOVehicleDetector(confidence_threshold=0.25)
            except Exception as e:
                print(f"Failed to load YOLO: {e}")
                self.use_yolo = False
        
        
    def detect_occupancy(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for slot in self.slots:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [slot.points], 255)
            
            roi_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            pixels = cv2.countNonZero(mask)
            if pixels == 0:
                slot.occupied = False
                continue
            
            # Calculate features
            mean_intensity = cv2.mean(roi_gray, mask=mask)[0]
            std_intensity = np.std(roi_gray[mask > 0])
            
            # Normalize intensity to 0-1 range
            normalized_intensity = mean_intensity / 255.0
            
            # Occupied spaces typically have:
            # - Lower standard deviation (uniform car surface)
            # - Variable mean intensity depending on car color
            # Empty spaces typically have:
            # - Higher standard deviation (parking lines, ground texture)
            # - More consistent brightness (usually lighter pavement)
            
            # Score based on uniformity - lower std = more likely occupied
            # Adjust threshold: if std is very low (< 25), likely occupied
            # If std is high (> 35), likely empty
            
            if std_intensity < 20:
                # Very uniform = empty pavement
                slot.occupied = False
            elif std_intensity > 40:
                # High variation = occupied (car details, shadows)
                slot.occupied = True
            else:
                # Medium range - use combined scoring
                # Higher std and more variation = more likely occupied
                variation_score = std_intensity / 100.0
                darkness_score = 1.0 - normalized_intensity
                
                occupancy_score = (variation_score * 0.7) + (darkness_score * 0.3)
                
                threshold = 0.4
                slot.occupied = occupancy_score > threshold
        
        return frame
    
    def detect_occupancy_yolo(self, frame):
        """
        Detect occupancy using YOLO vehicle detection (AI-based)
        """
        if not self.yolo_detector:
            return self.detect_occupancy(frame)  # Fallback to classic method
        
        # Detect vehicles using YOLO
        detections = self.yolo_detector.detect_vehicles(frame)
        
        # Reset all slots to empty
        for slot in self.slots:
            slot.occupied = False
        
        # Check each detection against parking slots
        for detection in detections:
            bbox = detection['bbox']
            
            # Check which slots this vehicle overlaps with
            for slot in self.slots:
                if self.yolo_detector.check_overlap(bbox, slot.points):
                    slot.occupied = True
        
        return frame
    
    def set_detection_mode(self, use_yolo):
        """Switch between YOLO and classic detection"""
        if use_yolo and self.yolo_detector is None:
            try:
                print("Loading YOLO detector...")
                # Lower confidence threshold to better detect stationary parked vehicles
                self.yolo_detector = YOLOVehicleDetector(confidence_threshold=0.25)
                self.use_yolo = True
            except Exception as e:
                print(f"Failed to load YOLO: {e}")
                self.use_yolo = False
                return False
        self.use_yolo = use_yolo
        return True
    
    
    def draw_slots(self, frame):
        overlay = frame.copy()
        
        for slot in self.slots:
            color = (0, 0, 255) if slot.occupied else (0, 255, 0)
            cv2.polylines(overlay, [slot.points], True, color, 2)
            cv2.fillPoly(overlay, [slot.points], color)
            
            M = cv2.moments(slot.points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(slot.slot_id), (cx - 10, cy + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    def get_stats(self):
        total = len(self.slots)
        occupied = sum(1 for slot in self.slots if slot.occupied)
        free = total - occupied
        return total, free, occupied
    
    def save_slots(self, filename):
        data = [slot.to_dict() for slot in self.slots]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_slots(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.slots = [ParkingSlot.from_dict(slot_data) for slot_data in data]
            return True
        return False
    
    def add_slot(self, points):
        slot_id = len(self.slots) + 1
        self.slots.append(ParkingSlot(points, slot_id))
    
    def clear_slots(self):
        self.slots = []


class CalibrationWindow:
    def __init__(self, parent, frame, detector, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("Calibration - Draw Parking Slots")
        self.window.geometry("1200x800")
        
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.detector = detector
        self.callback = callback
        
        self.drawing = False
        self.draw_mode = 'rectangle'
        self.current_points = []
        self.temp_slots = []
        
        self.scale_factor = 1.0
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        control_frame = ttk.Frame(self.window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value='rectangle')
        ttk.Radiobutton(control_frame, text="Rectangle", variable=self.mode_var,
                       value='rectangle', command=self.change_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Polygon", variable=self.mode_var,
                       value='polygon', command=self.change_mode).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Undo Last", command=self.undo_last).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save & Exit", command=self.save_and_exit).pack(side=tk.LEFT, padx=20)
        
        self.canvas_frame = ttk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def change_mode(self):
        self.draw_mode = self.mode_var.get()
        self.current_points = []
        
    def scale_point(self, x, y):
        return int(x / self.scale_factor), int(y / self.scale_factor)
    
    def on_mouse_down(self, event):
        x, y = self.scale_point(event.x, event.y)
        
        if self.draw_mode == 'rectangle':
            self.drawing = True
            self.current_points = [(x, y)]
        elif self.draw_mode == 'polygon':
            self.current_points.append((x, y))
            if len(self.current_points) > 1:
                self.update_display()
    
    def on_mouse_move(self, event):
        if self.draw_mode == 'rectangle' and self.drawing:
            x, y = self.scale_point(event.x, event.y)
            if len(self.current_points) == 1:
                self.current_points.append((x, y))
            else:
                self.current_points[1] = (x, y)
            self.update_display()
    
    def on_mouse_up(self, event):
        if self.draw_mode == 'rectangle' and self.drawing:
            x, y = self.scale_point(event.x, event.y)
            if len(self.current_points) == 1:
                self.current_points.append((x, y))
            else:
                self.current_points[1] = (x, y)
            
            if abs(self.current_points[1][0] - self.current_points[0][0]) > 10 and \
               abs(self.current_points[1][1] - self.current_points[0][1]) > 10:
                x1, y1 = self.current_points[0]
                x2, y2 = self.current_points[1]
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                self.temp_slots.append(points)
            
            self.current_points = []
            self.drawing = False
            self.update_display()
    
    def update_display(self):
        display = self.original_frame.copy()
        
        for points in self.temp_slots:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)
        
        if len(self.current_points) > 0:
            if self.draw_mode == 'rectangle' and len(self.current_points) == 2:
                x1, y1 = self.current_points[0]
                x2, y2 = self.current_points[1]
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
            elif self.draw_mode == 'polygon':
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(display, [pts], False, (255, 255, 0), 2)
                for pt in self.current_points:
                    cv2.circle(display, pt, 3, (0, 0, 255), -1)
        
        self.display_frame = display
        self.show_frame()
    
    def show_frame(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.window.after(100, self.show_frame)
            return
        
        h, w = self.display_frame.shape[:2]
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        self.scale_factor = min(scale_w, scale_h)
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        resized = cv2.resize(self.display_frame, (new_w, new_h))
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def clear_all(self):
        self.temp_slots = []
        self.current_points = []
        self.update_display()
    
    def undo_last(self):
        if self.temp_slots:
            self.temp_slots.pop()
            self.update_display()
    
    def save_and_exit(self):
        if self.draw_mode == 'polygon' and len(self.current_points) >= 3:
            self.temp_slots.append(self.current_points.copy())
            self.current_points = []
        
        for points in self.temp_slots:
            self.detector.add_slot(points)
        
        self.callback()
        self.window.destroy()


class AutoDetectWindow:
    def __init__(self, parent, frame, detector, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("Auto Detect Parking Slots")
        self.window.geometry("1200x850")
        
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.detector = detector
        self.callback = callback
        
        self.scale_factor = 1.0
        self.temp_slots = []
        
        # Default parameters
        self.start_x = 50
        self.start_y = 50
        self.slot_width = 50
        self.slot_height = 80
        self.rows = 3
        self.cols = 5
        self.spacing_x = 10
        self.spacing_y = 10
        self.angle = 0
        
        self.setup_ui()
        self.generate_grid()
        
    def setup_ui(self):
        # Control panel
        control_frame = ttk.Frame(self.window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Instructions
        ttk.Label(control_frame, text="Adjust parameters to generate parking slot grid:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=4, pady=5, sticky=tk.W)
        
        # Start position
        ttk.Label(control_frame, text="Start X:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.start_x_scale = tk.Scale(control_frame, from_=0, to=1000, orient=tk.HORIZONTAL,
                                     command=lambda v: self.update_param('start_x', int(v)))
        self.start_x_scale.set(self.start_x)
        self.start_x_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(control_frame, text="Start Y:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.start_y_scale = tk.Scale(control_frame, from_=0, to=1000, orient=tk.HORIZONTAL,
                                     command=lambda v: self.update_param('start_y', int(v)))
        self.start_y_scale.set(self.start_y)
        self.start_y_scale.grid(row=1, column=3, sticky=tk.EW, padx=5)
        
        # Slot dimensions
        ttk.Label(control_frame, text="Slot Width:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.width_scale = tk.Scale(control_frame, from_=20, to=200, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_param('slot_width', int(v)))
        self.width_scale.set(self.slot_width)
        self.width_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(control_frame, text="Slot Height:").grid(row=2, column=2, sticky=tk.W, padx=5)
        self.height_scale = tk.Scale(control_frame, from_=30, to=250, orient=tk.HORIZONTAL,
                                    command=lambda v: self.update_param('slot_height', int(v)))
        self.height_scale.set(self.slot_height)
        self.height_scale.grid(row=2, column=3, sticky=tk.EW, padx=5)
        
        # Grid layout
        ttk.Label(control_frame, text="Rows:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.rows_scale = tk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                  command=lambda v: self.update_param('rows', int(v)))
        self.rows_scale.set(self.rows)
        self.rows_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(control_frame, text="Columns:").grid(row=3, column=2, sticky=tk.W, padx=5)
        self.cols_scale = tk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                  command=lambda v: self.update_param('cols', int(v)))
        self.cols_scale.set(self.cols)
        self.cols_scale.grid(row=3, column=3, sticky=tk.EW, padx=5)
        
        # Spacing
        ttk.Label(control_frame, text="X Spacing:").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.spacing_x_scale = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                       command=lambda v: self.update_param('spacing_x', int(v)))
        self.spacing_x_scale.set(self.spacing_x)
        self.spacing_x_scale.grid(row=4, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(control_frame, text="Y Spacing:").grid(row=4, column=2, sticky=tk.W, padx=5)
        self.spacing_y_scale = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                       command=lambda v: self.update_param('spacing_y', int(v)))
        self.spacing_y_scale.set(self.spacing_y)
        self.spacing_y_scale.grid(row=4, column=3, sticky=tk.EW, padx=5)
        
        # Angle
        ttk.Label(control_frame, text="Angle (degrees):").grid(row=5, column=0, sticky=tk.W, padx=5)
        self.angle_scale = tk.Scale(control_frame, from_=-45, to=45, orient=tk.HORIZONTAL,
                                   command=lambda v: self.update_param('angle', int(v)))
        self.angle_scale.set(self.angle)
        self.angle_scale.grid(row=5, column=1, sticky=tk.EW, padx=5)
        
        # Configure column weights
        for i in range(4):
            control_frame.columnconfigure(i, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=5, column=2, columnspan=2, pady=5, sticky=tk.E)
        
        ttk.Button(button_frame, text="Clear All", command=self.clear_slots).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save & Exit", command=self.save_and_exit).pack(side=tk.LEFT, padx=5)
        
        # Canvas
        self.canvas_frame = ttk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def update_param(self, param, value):
        setattr(self, param, value)
        self.generate_grid()
        
    def generate_grid(self):
        self.temp_slots = []
        
        for row in range(self.rows):
            for col in range(self.cols):
                x = self.start_x + col * (self.slot_width + self.spacing_x)
                y = self.start_y + row * (self.slot_height + self.spacing_y)
                
                # Create rectangle points
                points = [
                    (x, y),
                    (x + self.slot_width, y),
                    (x + self.slot_width, y + self.slot_height),
                    (x, y + self.slot_height)
                ]
                
                # Apply rotation if angle is set
                if self.angle != 0:
                    angle_rad = np.radians(self.angle)
                    cx = x + self.slot_width / 2
                    cy = y + self.slot_height / 2
                    
                    rotated_points = []
                    for px, py in points:
                        # Translate to origin
                        tx = px - cx
                        ty = py - cy
                        # Rotate
                        rx = tx * np.cos(angle_rad) - ty * np.sin(angle_rad)
                        ry = tx * np.sin(angle_rad) + ty * np.cos(angle_rad)
                        # Translate back
                        rotated_points.append((int(rx + cx), int(ry + cy)))
                    points = rotated_points
                
                self.temp_slots.append(points)
        
        self.update_display()
    
    def update_display(self):
        display = self.original_frame.copy()
        
        for points in self.temp_slots:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)
            cv2.fillPoly(display, [pts], (0, 255, 255))
        
        # Blend
        if len(self.temp_slots) > 0:
            display = cv2.addWeighted(self.original_frame, 0.6, display, 0.4, 0)
        
        # Add slot count text
        cv2.putText(display, f"Total Slots: {len(self.temp_slots)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.display_frame = display
        self.show_frame()
    
    def show_frame(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.window.after(100, self.show_frame)
            return
        
        h, w = self.display_frame.shape[:2]
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        self.scale_factor = min(scale_w, scale_h)
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        resized = cv2.resize(self.display_frame, (new_w, new_h))
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def clear_slots(self):
        self.temp_slots = []
        self.update_display()
    
    def save_and_exit(self):
        for points in self.temp_slots:
            self.detector.add_slot(points)
        
        self.callback()
        self.window.destroy()


class LineDetectWindow:
    """Window for line-based automatic slot detection"""
    def __init__(self, parent, frame, detector, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("Auto Detect from Lines")
        self.window.geometry("1200x850")
        
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.detector = detector
        self.callback = callback
        
        self.line_detector = LineBasedSlotDetector()
        self.detected_slots = []
        self.thresh_image = None
        self.show_analysis = False
        
        self.setup_ui()
        self.detect_cars()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for displaying frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_panel = ttk.Frame(main_frame, width=250)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        ttk.Label(control_panel, text="Smart Auto Detection",
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # View toggle
        ttk.Button(control_panel, text="Toggle Analysis View",
                  command=self.toggle_view).pack(pady=5, fill=tk.X)
        
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Info
        info_text = ("🤖 AI-powered detection\n\n"
                    "Automatically finds cars and\n"
                    "generates parking slots.\n\n"
                    "Works for aerial parking lots!")
        ttk.Label(control_panel, text=info_text, wraplength=230,
                 justify=tk.LEFT).pack(pady=10)
        
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Buttons
        ttk.Button(control_panel, text="🔄 Re-detect",
                  command=self.detect_cars).pack(pady=5, fill=tk.X)
        
        ttk.Button(control_panel, text="✓ Accept Slots",
                  command=self.save_and_exit).pack(pady=20, fill=tk.X)
        
        ttk.Button(control_panel, text="Cancel",
                  command=self.window.destroy).pack(pady=5, fill=tk.X)
    
    def detect_cars(self):
        """Detect cars and generate slots automatically"""
        slots, thresh = self.line_detector.detect_cars_and_slots(self.original_frame)
        
        self.detected_slots = slots
        self.thresh_image = thresh
        
        if len(slots) == 0:
            messagebox.showwarning("No Cars Found",
                                 "Could not detect cars automatically.\n"
                                 "Try using grid-based detection instead.")
        else:
            messagebox.showinfo("Cars Detected!",
                              f"Found {len(slots)} parking spaces!\n\n"
                              f"Click 'Accept Slots' to use them.")
        
        self.update_display()
    
    def toggle_view(self):
        """Toggle between showing slots and analysis"""
        self.show_analysis = not self.show_analysis
        self.update_display()
    
    def update_display(self):
        """Update the display based on current state"""
        if self.show_analysis and self.thresh_image is not None:
            display = self.line_detector.draw_detected_cars(
                self.original_frame, self.thresh_image)
        else:
            if self.detected_slots:
                display = self.line_detector.draw_generated_slots(
                    self.original_frame, self.detected_slots)
            else:
                display = self.original_frame.copy()
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = display.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(display, (new_w, new_h))
            
            # Convert to PhotoImage
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2,
                                    image=photo, anchor=tk.CENTER)
            self.canvas.image = photo
    
    def save_and_exit(self):
        """Save generated slots to detector"""
        if not self.detected_slots:
            messagebox.showwarning("No Slots",
                                 "No slots detected! Try re-detecting.")
            return
        
        for slot_points in self.detected_slots:
            self.detector.add_slot(slot_points)
        
        self.callback()
        self.window.destroy()


class SmartParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Parking System")
        self.root.geometry("1400x900")
        
        self.detector = ParkingDetector()
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.calibration_frame = None
        self.reference_image = None  # Empty lot image for slot detection
        self.homography_matrix = None  # Perspective transform from reference to video
        self.yolo_detector = None
        self.is_detecting = False
        self.is_playing = False
        self.current_mode = "idle"
        
        self.slots_file = "parking_slots.json"
        # self.detector.load_slots(self.slots_file)  # Disabled - manual only
        
        # Modern UI styling
        self.setup_modern_style()

        
        self.setup_ui()
        self.update_stats()
    
    def setup_modern_style(self):
        """Setup premium dark UI theme - sleek and professional"""
        style = ttk.Style()
        style.theme_use('default')
        
        # Premium dark color palette - Black with electric blue accents
        self.bg_main = '#000000'           # Pure black
        self.bg_sidebar = '#0a0a0a'        # Slightly lighter black
        self.bg_card = '#121212'           # Dark gray
        self.border_color = '#1e1e1e'      # Subtle border
        self.accent_primary = '#00d4ff'    # Electric blue
        self.accent_success = '#00ff88'    # Bright green
        self.accent_danger = '#ff3366'     # Vibrant red
        self.text_primary = '#ffffff'      # Pure white
        self.text_secondary = '#888888'    # Medium gray
        
        # Configure styles
        style.configure('TFrame', background=self.bg_main)
        style.configure('Card.TFrame', background=self.bg_card, relief='flat')
        
        style.configure('TLabel',
                       background=self.bg_main,
                       foreground=self.text_primary,
                       font=('Segoe UI', 11))
        
        # Modern button styles
        style.configure('Modern.TButton',
                       font=('Segoe UI', 10),
                       borderwidth=1,
                       relief='flat',
                       padding=(14, 10))
        
        style.map('Modern.TButton',
                 background=[('active', '#1a1a1a'), ('!disabled', self.bg_card)],
                 foreground=[('active', self.text_primary), ('!disabled', self.text_secondary)],
                 bordercolor=[('!disabled', self.border_color)])
        
        style.configure('Primary.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       padding=(14, 10))
        
        style.map('Primary.TButton',
                 background=[('active', '#00a8cc'), ('!disabled', self.accent_primary)],
                 foreground=[('active', '#000000'), ('!disabled', '#000000')])
        
        style.configure('Success.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       padding=(14, 10))
        
        style.map('Success.TButton',
                 background=[('active', '#00cc66'), ('!disabled', self.accent_success)],
                 foreground=[('active', '#000000'), ('!disabled', '#000000')])
        
        style.configure('Danger.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       padding=(14, 10))
        
        style.map('Danger.TButton',
                 background=[('active', '#cc2244'), ('!disabled', self.accent_danger)],
                 foreground=[('active', 'white'), ('!disabled', 'white')])
        
        # LabelFrame styling
        style.configure('Card.TLabelframe',
                       background=self.bg_card,
                       borderwidth=0,
                       relief='flat')
        
        style.configure('Card.TLabelframe.Label',
                       background=self.bg_card,
                       foreground=self.text_primary,
                       font=('Segoe UI', 12, 'bold'))
        
        self.root.configure(bg=self.bg_main)
    
    def setup_ui(self):
        """Setup clean, modern, scrollable UI"""
        # Header
        header = tk.Frame(self.root, bg=self.bg_sidebar, height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="SMART PARKING", 
                        font=('Segoe UI', 28, 'bold'),
                        bg=self.bg_sidebar, fg=self.text_primary)
        title.pack(side=tk.LEFT, padx=20, pady=15)
        
        badge = tk.Label(header, text="AI", 
                        font=('Segoe UI', 11, 'bold'),
                        bg=self.accent_primary, fg='#000000',
                        padx=8, pady=3)
        badge.pack(side=tk.LEFT)
        
        # Main container
        main = tk.Frame(self.root, bg=self.bg_main)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Left sidebar with canvas for scrolling
        sidebar_container = tk.Frame(main, bg=self.bg_sidebar, width=280, relief='flat', bd=0)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        sidebar_container.pack_propagate(False)
        
        # Create canvas and scrollbar for sidebar
        canvas = tk.Canvas(sidebar_container, bg=self.bg_sidebar, highlightthickness=0)
        
        # Configure scrollbar to be VISIBLE on dark theme
        scrollbar = tk.Scrollbar(sidebar_container, orient="vertical", command=canvas.yview,
                                bg=self.accent_primary,  # Electric blue
                                troughcolor=self.bg_card,  # Dark gray
                                activebackground='white',  # White when hovering
                                width=12)  # Wider for visibility
        
        scrollable_frame = tk.Frame(canvas, bg=self.bg_sidebar)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Sidebar content - MINIMAL
        pad = 12
        
        # Stats
        tk.Label(scrollable_frame, text="STATS", font=('Segoe UI', 9, 'bold'),
                bg=self.bg_sidebar, fg=self.text_secondary).pack(padx=pad, pady=(10, 4), anchor=tk.W)
        
        self.total_label = tk.Label(scrollable_frame, text="Total: 0",
                                   bg=self.bg_sidebar, fg=self.text_primary,
                                   font=('Segoe UI', 10, 'bold'))
        self.total_label.pack(padx=pad, anchor=tk.W)
        
        self.free_label = tk.Label(scrollable_frame, text="Free: 0",
                                  bg=self.bg_sidebar, fg=self.accent_success,
                                  font=('Segoe UI', 10, 'bold'))
        self.free_label.pack(padx=pad, anchor=tk.W)
        
        self.occupied_label = tk.Label(scrollable_frame, text="Occupied: 0",
                                      bg=self.bg_sidebar, fg=self.accent_danger,
                                      font=('Segoe UI', 10, 'bold'))
        self.occupied_label.pack(padx=pad, pady=(0,6), anchor=tk.W)
        
        tk.Frame(scrollable_frame, bg=self.border_color, height=1).pack(fill=tk.X, padx=pad, pady=4)
        
        # Setup
        ttk.Button(scrollable_frame, text="Load Video",
                  command=self.load_video,
                  style='Modern.TButton').pack(padx=pad, pady=3, fill=tk.X)
        
        self.capture_btn = ttk.Button(scrollable_frame, text="Capture Frame",
                                     command=self.capture_frame, state=tk.DISABLED,
                                     style='Modern.TButton')
        self.capture_btn.pack(padx=pad, pady=3, fill=tk.X)
        
        tk.Frame(scrollable_frame, bg=self.border_color, height=1).pack(fill=tk.X, padx=pad, pady=4)
        
        # Detection - Manual Only
        self.draw_btn = ttk.Button(scrollable_frame, text="✏️ Draw Slots Manually",
                                  command=self.open_calibration, state=tk.DISABLED,
                                  style='Primary.TButton')
        self.draw_btn.pack(padx=pad, pady=3, fill=tk.X)
        
        tk.Frame(scrollable_frame, bg=self.border_color, height=1).pack(fill=tk.X, padx=pad, pady=4)

        
        # Detection mode
        self.detection_mode_var = tk.StringVar(value='classic')
        
        tk.Label(scrollable_frame, text="MODE", font=('Segoe UI', 9, 'bold'),
                bg=self.bg_sidebar, fg=self.text_secondary).pack(padx=pad, pady=(4, 2), anchor=tk.W)
        
        mode_frame = tk.Frame(scrollable_frame, bg=self.bg_sidebar)
        mode_frame.pack(padx=pad, pady=(0, 4), fill=tk.X)
        
        tk.Radiobutton(mode_frame, text="Classic", variable=self.detection_mode_var, 
                      value='classic', command=self.change_detection_mode,
                      bg=self.bg_sidebar, fg=self.text_primary, selectcolor=self.bg_sidebar,
                      font=('Segoe UI', 9), activebackground=self.bg_sidebar).pack(anchor=tk.W)
        
        tk.Radiobutton(mode_frame, text="YOLO", variable=self.detection_mode_var,
                      value='yolo', command=self.change_detection_mode,
                      bg=self.bg_sidebar, fg=self.text_primary, selectcolor=self.bg_sidebar,
                      font=('Segoe UI', 9), activebackground=self.bg_sidebar).pack(anchor=tk.W)
        
        tk.Frame(scrollable_frame, bg=self.border_color, height=1).pack(fill=tk.X, padx=pad, pady=4)
        
        # START/STOP
        self.start_btn = ttk.Button(scrollable_frame, text="▶ START",
                                   command=self.start_detection, state=tk.DISABLED,
                                   style='Success.TButton')
        self.start_btn.pack(padx=pad, pady=3, fill=tk.X)
        
        self.stop_btn = ttk.Button(scrollable_frame, text="⏹ STOP",
                                  command=self.stop_detection, state=tk.DISABLED,
                                  style='Danger.TButton')
        self.stop_btn.pack(padx=pad, pady=3, fill=tk.X)
        
        tk.Frame(scrollable_frame, bg=self.border_color, height=1).pack(fill=tk.X, padx=pad, pady=4)
        
        # EXIT
        ttk.Button(scrollable_frame, text="❌ EXIT",
                  command=self.exit_app,
                  style='Danger.TButton').pack(padx=pad, pady=3, fill=tk.X)

        
        # Right panel - Video
        video_panel = tk.Frame(main, bg=self.bg_main)
        video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        video_label = tk.Label(video_panel, text="LIVE FEED",
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.bg_main, fg=self.accent_primary)
        video_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Canvas with subtle border
        canvas_frame = tk.Frame(video_panel, bg=self.border_color, padx=1, pady=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status = tk.Frame(self.root, bg=self.bg_sidebar, height=30)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        status.pack_propagate(False)
        
        self.mode_label = tk.Label(status, text="● Ready",
                                  bg=self.bg_sidebar, fg=self.accent_success,
                                  font=('Segoe UI', 10))
        self.mode_label.pack(side=tk.LEFT, padx=20, pady=5)
    
    def _create_section(self, parent, title, padx):
        """Helper to create section header"""
        label = tk.Label(parent, text=title,
                        font=('Segoe UI', 11, 'bold'),
                        bg=self.bg_sidebar, fg=self.text_secondary)
        label.pack(anchor=tk.W, padx=padx, pady=(12, 8))
    
    def _create_divider(self, parent):
        """Helper to create subtle divider"""
        div = tk.Frame(parent, bg=self.border_color, height=1)
        div.pack(fill=tk.X, padx=15, pady=8)

        
    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        
        if filepath:
            self.video_path = filepath
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            
            if self.cap.isOpened():
                messagebox.showinfo("Success", "Video loaded successfully!")
                self.capture_btn.config(state=tk.NORMAL)
                if len(self.detector.slots) > 0:
                    self.start_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Failed to load video!")
    
    def capture_frame(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.calibration_frame = frame.copy()
                self.draw_btn.config(state=tk.NORMAL)
                messagebox.showinfo("Success", "Calibration frame captured! Choose a slot detection method")
                self.set_mode("calibration")
    
    def load_reference_image(self):
        """Load empty parking lot reference image for better slot detection"""
        filepath = filedialog.askopenfilename(
            title="Select Empty Parking Lot Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        
        if filepath:
            ref_img = cv2.imread(filepath)
            if ref_img is not None:
                self.reference_image = ref_img
                messagebox.showinfo("Success", 
                                  "Empty lot image loaded!\n\n"
                                  "Now auto-detecting parking slots...")
                
                # Auto-detect slots from empty lot
                LineDetectWindow(
                    self.root,
                    self.reference_image.copy(),
                    self.detector,
                    lambda: self.update_stats()
                )
            else:
                messagebox.showerror("Error", "Failed to load image!")
    
    def open_calibration(self):
        if self.calibration_frame is not None:
            CalibrationWindow(self.root, self.calibration_frame, self.detector, self.on_calibration_complete)
    
    def open_auto_detect(self):
        if self.calibration_frame is not None:
            AutoDetectWindow(self.root, self.calibration_frame, self.detector, self.on_calibration_complete)
    
    def open_line_detect(self):
        """Open line-based automatic detection window"""
        if self.calibration_frame is not None:
            LineDetectWindow(self.root, self.calibration_frame, self.detector, self.on_calibration_complete)
    
    def on_calibration_complete(self):
        self.update_stats()
        if len(self.detector.slots) > 0:
            self.start_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", f"{len(self.detector.slots)} parking slots defined!")
    
    def save_slots(self):
        self.detector.save_slots(self.slots_file)
        messagebox.showinfo("Success", f"Parking slots saved to {self.slots_file}")
    
    def reset_slots(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to reset all parking slots?"):
            self.detector.clear_slots()
            self.update_stats()
            self.start_btn.config(state=tk.DISABLED)
            messagebox.showinfo("Success", "All parking slots cleared!")
    
    
    def calculate_alignment(self, reference_img, video_frame):
        """
        Calculate homography matrix to align reference image with video frame
        Uses feature matching to find perspective transformation
        """
        try:
            # Convert to grayscale
            ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            vid_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB features
            orb = cv2.ORB_create(5000)
            kp1, des1 = orb.detectAndCompute(ref_gray, None)
            kp2, des2 = orb.detectAndCompute(vid_gray, None)
            
            if des1 is None or des2 is None:
                print("⚠️ Could not detect features for alignment")
                return None
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                print(f"⚠️ Not enough good matches: {len(good_matches)}")
                return None
            
            # Get matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Calculate homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            inliers = np.sum(mask) if mask is not None else 0
            print(f"✅ Alignment: {len(good_matches)} matches, {inliers} inliers")
            return H
        except Exception as e:
            print(f"❌ Alignment failed: {e}")
            return None
    
    def transform_slots(self, slots, homography):
        """
        Transform slot coordinates using homography matrix
        """
        if homography is None or len(slots) == 0:
            return slots
        
        try:
            transformed_slots = []
            for slot in slots:
                # Convert slot to numpy array
                pts = np.array(slot, dtype=np.float32).reshape(-1, 1, 2)
                
                # Apply perspective transform
                transformed_pts = cv2.perspectiveTransform(pts, homography)
                
                # Convert back to list of tuples
                transformed_slot = [(int(pt[0][0]), int(pt[0][1])) for pt in transformed_pts]
                transformed_slots.append(transformed_slot)
            
            return transformed_slots
        except Exception as e:
            print(f"❌ Slot transformation failed: {e}")
            return slots
    
    def start_detection(self):
        if self.cap and len(self.detector.slots) > 0:
            self.is_playing = True
            self.set_mode("detection")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.DISABLED)
            self.draw_btn.config(state=tk.DISABLED)
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            threading.Thread(target=self.detection_loop, daemon=True).start()
    
    def stop_detection(self):
        self.is_playing = False
        self.set_mode("idle")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
        self.draw_btn.config(state=tk.NORMAL if self.calibration_frame is not None else tk.DISABLED)
    
    def detection_loop(self):
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Use appropriate detection method
            if self.detector.use_yolo:
                self.detector.detect_occupancy_yolo(frame)
            else:
                self.detector.detect_occupancy(frame)
                
            display_frame = self.detector.draw_slots(frame)
            
            self.display_frame(display_frame)
            self.update_stats()
            
            cv2.waitKey(30)
        
        if not self.is_playing:
            self.root.after(0, self.update_stats)
    
    def display_frame(self, frame):
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            h, w = frame.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
        except:
            pass
    
    def update_stats(self):
        total, free, occupied = self.detector.get_stats()
        self.total_label.config(text=f"Total: {total}")
        self.free_label.config(text=f"Free: {free}")
        self.occupied_label.config(text=f"Occupied: {occupied}")
    
    def change_detection_mode(self):
        """Handle detection mode changes"""
        mode = self.detection_mode_var.get()
        use_yolo = (mode == 'yolo')
        
        if use_yolo:
            # Try to enable YOLO
            if not self.detector.set_detection_mode(True):
                messagebox.showerror("Error", 
                                   "Failed to load YOLO model. Please ensure ultralytics is installed correctly.")
                self.detection_mode_var.set('classic')
            else:
                messagebox.showinfo("Success", 
                                  "AI Smart Detection (YOLO) enabled! This uses machine learning to detect vehicles.")
        else:
            self.detector.set_detection_mode(False)
            messagebox.showinfo("Detection Mode", 
                              "Classic Detection enabled (texture-based algorithm)")
    
    def set_mode(self, mode):
        self.current_mode = mode
        mode_text = {
            "idle": "Mode: Idle",
            "calibration": "Mode: Calibration",
            "detection": "Mode: Detection Running"
        }
        self.mode_label.config(text=mode_text.get(mode, "Mode: Unknown"))
    
    def exit_app(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartParkingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()