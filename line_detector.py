"""
Line-based parking slot detection using computer vision
Detects parking lines and automatically generates slot polygons
"""

import cv2
import numpy as np
from collections import defaultdict


class LineBasedSlotDetector:
    def __init__(self):
        self.detected_lines = []
        self.slots = []
        
    def detect_parking_lines(self, frame):
        """
        Detect parking lines using Hough Line Transform
        
        Args:
            frame: Input calibration frame
            
        Returns:
            List of detected lines and processed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return [], edges
        
        # Filter and classify lines
        self.detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Keep mostly horizontal or vertical lines
            # Horizontal: angle near 0 or 180
            # Vertical: angle near 90
            if angle < 15 or angle > 165:  # Horizontal
                line_type = 'horizontal'
            elif 75 < angle < 105:  # Vertical
                line_type = 'vertical'
            else:
                continue  # Skip diagonal lines
            
            self.detected_lines.append({
                'coords': (x1, y1, x2, y2),
                'type': line_type,
                'angle': angle
            })
        
        return self.detected_lines, edges
    
    def generate_slots_from_lines(self, frame_shape, min_slots=4, slot_width_range=(50, 200)):
        """
        Generate parking slot polygons from detected lines
        
        Args:
            frame_shape: Shape of the frame (height, width)
            min_slots: Minimum number of slots to detect
            slot_width_range: Expected slot width range (min, max)
            
        Returns:
            List of slot polygons
        """
        if len(self.detected_lines) < 4:
            return []
        
        # Separate horizontal and vertical lines
        h_lines = [l for l in self.detected_lines if l['type'] == 'horizontal']
        v_lines = [l for l in self.detected_lines if l['type'] == 'vertical']
        
        slots = []
        
        # Method 1: Use vertical lines to define slot boundaries (perpendicular parking)
        if len(v_lines) >= 2:
            slots = self._generate_slots_perpendicular(v_lines, h_lines, frame_shape)
        
        # Method 2: Use horizontal lines (parallel parking)
        if len(slots) < min_slots and len(h_lines) >= 2:
            slots = self._generate_slots_parallel(h_lines, v_lines, frame_shape)
        
        self.slots = slots
        return slots
    
    def _generate_slots_perpendicular(self, v_lines, h_lines, frame_shape):
        """Generate slots for perpendicular parking (vertical dividers)"""
        slots = []
        height, width = frame_shape[:2]
        
        # Sort vertical lines by x-coordinate
        v_lines_sorted = sorted(v_lines, key=lambda l: (l['coords'][0] + l['coords'][2]) / 2)
        
        # Find horizontal boundaries (top and bottom of parking area)
        if h_lines:
            h_coords = [(l['coords'][1] + l['coords'][3]) / 2 for l in h_lines]
            top_y = int(min(h_coords))
            bottom_y = int(max(h_coords))
        else:
            # Use frame boundaries
            top_y = int(height * 0.2)
            bottom_y = int(height * 0.8)
        
        # Create slots between consecutive vertical lines
        for i in range(len(v_lines_sorted) - 1):
            x1 = int((v_lines_sorted[i]['coords'][0] + v_lines_sorted[i]['coords'][2]) / 2)
            x2 = int((v_lines_sorted[i+1]['coords'][0] + v_lines_sorted[i+1]['coords'][2]) / 2)
            
            # Skip if slot too narrow or too wide
            slot_width = abs(x2 - x1)
            if slot_width < 40 or slot_width > 250:
                continue
            
            # Create rectangle
            slot = [
                (x1, top_y),
                (x2, top_y),
                (x2, bottom_y),
                (x1, bottom_y)
            ]
            slots.append(slot)
        
        return slots
    
    def _generate_slots_parallel(self, h_lines, v_lines, frame_shape):
        """Generate slots for parallel parking (horizontal dividers)"""
        slots = []
        height, width = frame_shape[:2]
        
        # Sort horizontal lines by y-coordinate
        h_lines_sorted = sorted(h_lines, key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
        
        # Find vertical boundaries (left and right of parking area)
        if v_lines:
            v_coords = [(l['coords'][0] + l['coords'][2]) / 2 for l in v_lines]
            left_x = int(min(v_coords))
            right_x = int(max(v_coords))
        else:
            left_x = int(width * 0.1)
            right_x = int(width * 0.9)
        
        # Create slots between consecutive horizontal lines
        for i in range(len(h_lines_sorted) - 1):
            y1 = int((h_lines_sorted[i]['coords'][1] + h_lines_sorted[i]['coords'][3]) / 2)
            y2 = int((h_lines_sorted[i+1]['coords'][1] + h_lines_sorted[i+1]['coords'][3]) / 2)
            
            # Skip if slot too narrow or too wide
            slot_height = abs(y2 - y1)
            if slot_height < 60 or slot_height > 300:
                continue
            
            # Create rectangle
            slot = [
                (left_x, y1),
                (right_x, y1),
                (right_x, y2),
                (left_x, y2)
            ]
            slots.append(slot)
        
        return slots
    
    def draw_detected_lines(self, frame):
        """Draw detected lines on frame for visualization"""
        display = frame.copy()
        
        for line in self.detected_lines:
            x1, y1, x2, y2 = line['coords']
            color = (0, 255, 0) if line['type'] == 'vertical' else (255, 0, 0)
            cv2.line(display, (x1, y1), (x2, y2), color, 2)
        
        # Add legend
        cv2.putText(display, "Green: Vertical | Blue: Horizontal", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    def draw_generated_slots(self, frame, slots):
        """Draw generated slots on frame"""
        display = frame.copy()
        
        for i, slot in enumerate(slots):
            pts = np.array(slot, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)
            
            # Add slot number
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(display, str(i+1), tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display, f"Detected {len(slots)} slots", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display
    
    def detect_cars_and_slots(self, frame):
        """
        STRICT detection: Only detects slots with clear WHITE parking lines
        Does NOT fill entire image - only where lines are visible
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect bright areas (very relaxed threshold 150)
        _, white_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Apply morphology to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Blur for better edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 30, 100)
        
        # Only keep edges that are on white areas
        edges = cv2.bitwise_and(edges, edges, mask=white_mask)
        
        # Detect lines - very lenient
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=25,       # Very low threshold
                               minLineLength=20,   # Short lines OK
                               maxLineGap=30)      # Large gaps OK
        
        if lines is None or len(lines) < 3:  # Only need 3 lines total
            return [], edges

        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Ignore very short lines
            if length < 30:
                continue
            
            # Calculate angle
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify by angle - must be very close to horizontal/vertical
            if angle < 10:  # Nearly horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle > 75:  # Nearly vertical (relaxed)
                vertical_lines.append((x1, y1, x2, y2))
        
        # Only need 2 horizontal lines minimum
        if len(horizontal_lines) < 2:
            return [], edges

        
        slots = []
        
        # Get Y positions of horizontal lines
        y_positions = []
        for x1, y1, x2, y2 in horizontal_lines:
            y_avg = (y1 + y2) // 2
            y_positions.append(y_avg)
        
        # Cluster nearby Y positions
        y_positions = sorted(y_positions)
        clustered_y = []
        
        if y_positions:
            current_cluster = [y_positions[0]]
            for y in y_positions[1:]:
                if y - current_cluster[-1] < 25:  # Merge very close lines
                    current_cluster.append(y)
                else:
                    clustered_y.append(int(np.mean(current_cluster)))
                    current_cluster = [y]
            clustered_y.append(int(np.mean(current_cluster)))
        
        # Only need 2 lines minimum now
        if len(clustered_y) < 2:
            return [], edges

        
        # Detect vertical column boundaries
        if len(vertical_lines) >= 2:
            x_positions = []
            for x1, y1, x2, y2 in vertical_lines:
                x_avg = (x1 + x2) // 2
                x_positions.append(x_avg)
            
            x_positions = sorted(set(x_positions))
            
            # Cluster X positions
            clustered_x = []
            if x_positions:
                current_cluster = [x_positions[0]]
                for x in x_positions[1:]:
                    if x - current_cluster[-1] < 50:  # More forgiving
                        current_cluster.append(x)
                    else:
                        clustered_x.append(int(np.mean(current_cluster)))
                        current_cluster = [x]
                clustered_x.append(int(np.mean(current_cluster)))
            
            # Create slots where we have both horizontal and vertical lines
            for i in range(len(clustered_x) - 1):
                x1 = clustered_x[i]
                x2 = clustered_x[i + 1]
                
                # More lenient width check
                if not (40 < (x2 - x1) < 300):
                    continue
                
                for j in range(len(clustered_y) - 1):
                    y1 = clustered_y[j]
                    y2 = clustered_y[j + 1]
                    
                    # More lenient height check
                    if not (20 < (y2 - y1) < 200):
                        continue
                    
                    # Verify this area has some white lines
                    roi = white_mask[y1:y2, x1:x2]
                    if roi.size > 0:
                        white_percentage = np.sum(roi > 0) / roi.size
                        
                        # Very lenient - just needs 3% white
                        if white_percentage > 0.03:
                            slot = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            slots.append(slot)
        else:
            # FALLBACK: No vertical lines, create full-width slots
            margin = 30
            for j in range(len(clustered_y) - 1):
                y1 = clustered_y[j]
                y2 = clustered_y[j + 1]
                
                if (y2 - y1) > 20:  # Minimum height
                    slot = [(margin, y1), (w - margin, y1), 
                          (w - margin, y2), (margin, y2)]
                    slots.append(slot)


        
        self.slots = slots
        return slots, edges

        """
        Optimized for empty parking lots with clear white lines
        Works perfectly for perpendicular/angled parking layouts
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance white lines
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Strong blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Canny edge detection - tuned for white lines on gray pavement
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=100,      # Lower threshold to catch more lines
                               minLineLength=50,   # Shorter minimum length
                               maxLineGap=20)      # Allow larger gaps
        
        if lines is None or len(lines) < 5:
            return [], edges
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify by angle
            if angle < 15:  # Nearly horizontal (0-15 degrees)
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle > 75:  # Nearly vertical (75-90 degrees)
                vertical_lines.append((x1, y1, x2, y2))
        
        slots = []
        
        # STRATEGY 1: Use horizontal lines (parking space dividers)
        if len(horizontal_lines) >= 3:
            # Get Y positions of horizontal lines
            y_positions = []
            for x1, y1, x2, y2 in horizontal_lines:
                y_avg = (y1 + y2) // 2
                y_positions.append(y_avg)
            
            # Sort and cluster nearby Y positions
            y_positions = sorted(y_positions)
            
            clustered_y = []
            if y_positions:
                current_cluster = [y_positions[0]]
                for y in y_positions[1:]:
                    if y - current_cluster[-1] < 30:  # Merge lines within 30px
                        current_cluster.append(y)
                    else:
                        clustered_y.append(int(np.mean(current_cluster)))
                        current_cluster = [y]
                clustered_y.append(int(np.mean(current_cluster)))
            
            # Create slots between consecutive horizontal lines
            if len(clustered_y) >= 2:
                # Detect vertical columns
                if len(vertical_lines) >= 2:
                    x_positions = []
                    for x1, y1, x2, y2 in vertical_lines:
                        x_avg = (x1 + x2) // 2
                        x_positions.append(x_avg)
                    
                    x_positions = sorted(set(x_positions))
                    
                    # Cluster X positions
                    clustered_x = []
                    if x_positions:
                        current_cluster = [x_positions[0]]
                        for x in x_positions[1:]:
                            if x - current_cluster[-1] < 30:
                                current_cluster.append(x)
                            else:
                                clustered_x.append(int(np.mean(current_cluster)))
                                current_cluster = [x]
                        clustered_x.append(int(np.mean(current_cluster)))
                    
                    # Generate slots for each column
                    for i in range(len(clustered_x) - 1):
                        x1 = clustered_x[i]
                        x2 = clustered_x[i + 1]
                        
                        for j in range(len(clustered_y) - 1):
                            y1 = clustered_y[j]
                            y2 = clustered_y[j + 1]
                            
                            # Only create slot if it's reasonable size
                            if (x2 - x1) > 40 and (y2 - y1) > 30:
                                slot = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                                slots.append(slot)
                else:
                    # No vertical lines - use full width, divide by horizontal lines
                    margin = 50
                    for j in range(len(clustered_y) - 1):
                        y1 = clustered_y[j]
                        y2 = clustered_y[j + 1]
                        
                        if (y2 - y1) > 30:  # Minimum height
                            slot = [(margin, y1), (w - margin, y1), 
                                  (w - margin, y2), (margin, y2)]
                            slots.append(slot)
        
        # STRATEGY 2: If Strategy 1 didn't work, use grid approach
        if len(slots) < 3 and len(vertical_lines) >= 2:
            x_positions = sorted([((x1 + x2) // 2) for x1, y1, x2, y2 in vertical_lines])
            
            # Standard slot dimensions
            slot_height = 60
            num_rows = max(1, h // (slot_height + 10))
            
            for i in range(len(x_positions) - 1):
                x1 = x_positions[i]
                x2 = x_positions[i + 1]
                
                if (x2 - x1) > 40:  # Reasonable width
                    for row in range(num_rows):
                        y1 = row * (slot_height + 10) + 10
                        y2 = y1 + slot_height
                        
                        if y2 < h:
                            slot = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            slots.append(slot)
        
        self.slots = slots
        return slots, edges

        """
        Smart detection: Works for both occupied and empty lots
        - Empty lot: Detects parking lines/structure
        - Occupied lot: Detects cars
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try to detect parking lines first (for empty lots)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=30, maxLineGap=10)
        
        slots = []
        
        # If we detected strong lines, use line-based detection (empty lot)
        if lines is not None and len(lines) > 20:
            # Separate into vertical and horizontal lines
            vertical_lines = []
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle > 80 or angle < 10:  # Vertical-ish
                    vertical_lines.append((x1, y1, x2, y2))
                elif 80 < angle < 100:  # Horizontal-ish
                    horizontal_lines.append((x1, y1, x2, y2))
            
            # Group vertical lines by X position
            if len(vertical_lines) >= 2:
                x_positions = sorted(set([x1 for x1, y1, x2, y2 in vertical_lines] + 
                                       [x2 for x1, y1, x2, y2 in vertical_lines]))
                
                # Cluster nearby X positions
                clustered_x = []
                if x_positions:
                    current_cluster = [x_positions[0]]
                    for x in x_positions[1:]:
                        if x - current_cluster[-1] < 20:  # Merge nearby lines
                            current_cluster.append(x)
                        else:
                            clustered_x.append(int(np.mean(current_cluster)))
                            current_cluster = [x]
                    clustered_x.append(int(np.mean(current_cluster)))
                
                # Generate slots from vertical divisions
                for i in range(len(clustered_x) - 1):
                    x1 = clustered_x[i]
                    x2 = clustered_x[i + 1]
                    
                    # Standard parking slot height
                    slot_height = int((x2 - x1) * 0.4)  # Aspect ratio ~2.5:1
                    
                    # Create multiple rows if image is tall enough
                    num_rows = max(1, h // (slot_height + 20))
                    for row in range(num_rows):
                        y1 = row * (slot_height + 10)
                        y2 = y1 + slot_height
                        
                        if y2 < h:
                            slot = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            slots.append(slot)
        
        # If line detection didn't work, fall back to car detection
        if len(slots) < 3:
            slots, thresh = self._detect_from_cars(frame, gray, blurred)
            self.slots = slots
            return slots, thresh if 'thresh' in locals() else edges
        
        self.slots = slots
        return slots, edges
    
    def _detect_from_cars(self, frame, gray, blurred):
        """Original car-based detection for occupied lots"""
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)
        
        thresh = cv2.medianBlur(thresh, 5)
        
        # Light erosion to separate cars
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.erode(thresh, kernel_erode, iterations=1)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel_dilate, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        car_contours = []
        min_area = 500
        max_area = 12000
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                if 0.3 < aspect_ratio < 4.0 and solidity > 0.5:
                    car_contours.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
        
        if len(car_contours) < 2:
            return [], thresh
        
        car_contours.sort(key=lambda c: (c['center'][1], c['center'][0]))
        
        rows = []
        if car_contours:
            current_row = [car_contours[0]]
            y_tolerance = 60
            
            for car in car_contours[1:]:
                if abs(car['center'][1] - current_row[0]['center'][1]) < y_tolerance:
                    current_row.append(car)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [car]
            
            if current_row:
                rows.append(current_row)
        
        slots = []
        for row in rows:
            row.sort(key=lambda c: c['center'][0])
            
            for car in row:
                x, y, w, h = car['bbox']
                padding = 3
                slot = [
                    (x - padding, y - padding),
                    (x + w + padding, y - padding),
                    (x + w + padding, y + h + padding),
                    (x - padding, y + h + padding)
                ]
                slots.append(slot)
        
        return slots, thresh

        """
        Smart detection: Find individual cars and generate precise slots
        Uses erosion to separate touching cars + histogram analysis to find columns
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)
        
        # Median blur to reduce noise
        thresh = cv2.medianBlur(thresh, 5)
        
        # CRITICAL: Use light erosion to separate touching cars (reduced iterations)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.erode(thresh, kernel_erode, iterations=1)  # Reduced from 2 to 1
        
        # Then dilate to restore car size
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel_dilate, iterations=2)  # Increased from 1 to 2
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size - more inclusive now
        car_contours = []
        min_area = 500     # Slightly reduced to catch smaller cars
        max_area = 12000   # Increased to catch larger cars/merged regions

        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Calculate solidity (filled area) to filter out hollow shapes
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Accept various aspect ratios (cars can be horizontal or vertical)
                # AND must have good solidity (solid objects, not hollow/partial)
                if 0.3 < aspect_ratio < 4.0 and solidity > 0.5:
                    car_contours.append({
                        'contour': cnt,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })

        
        if len(car_contours) < 2:
            return [], thresh
        
        # Sort all cars by position (top to bottom, left to right)
        car_contours.sort(key=lambda c: (c['center'][1], c['center'][0]))
        
        # Group into rows based on Y coordinate
        rows = []
        if car_contours:
            current_row = [car_contours[0]]
            y_tolerance = 60  # Increased tolerance to catch more rows
            
            for car in car_contours[1:]:
                if abs(car['center'][1] - current_row[0]['center'][1]) < y_tolerance:
                    current_row.append(car)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [car]
            
            if current_row:
                rows.append(current_row)
        
        # Generate slots from each row
        slots = []
        for row in rows:
            # Sort row by X coordinate (left to right)
            row.sort(key=lambda c: c['center'][0])
            
            # Generate slots only for detected cars (no gap filling)
            for car in row:
                x, y, w, h = car['bbox']
                
                # Create tight slot around car
                padding = 3
                slot = [
                    (x - padding, y - padding),
                    (x + w + padding, y - padding),
                    (x + w + padding, y + h + padding),
                    (x - padding, y + h + padding)
                ]
                slots.append(slot)

        
        self.slots = slots
        return slots, thresh

    
    def draw_detected_cars(self, frame, thresh_img):
        """Draw detected cars and analysis"""
        display = frame.copy()
        
        # Show threshold image in corner
        h, w = thresh_img.shape
        scale = 0.25
        small_thresh = cv2.resize(thresh_img, (int(w*scale), int(h*scale)))
        small_thresh_bgr = cv2.cvtColor(small_thresh, cv2.COLOR_GRAY2BGR)
        
        th, tw = small_thresh_bgr.shape[:2]
        display[10:10+th, 10:10+tw] = small_thresh_bgr
        
        cv2.putText(display, "Car Detection Analysis", (10, 10+th+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display

