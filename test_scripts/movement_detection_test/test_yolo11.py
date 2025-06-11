import cv2
import numpy as np
from collections import defaultdict, deque
import time
from ultralytics import YOLO
import torch

class EnhancedMovementDetector:
    def __init__(self):
        # Use multiple detection methods for better accuracy
        self.setup_background_subtractors()
        self.setup_detection_parameters()
        self.setup_tracking()
        
    def setup_background_subtractors(self):
        """Initialize multiple background subtraction methods"""
        # MOG2 - good for general scenarios
        self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,  # Lower threshold for better sensitivity
            detectShadows=True
        )
        
        # KNN - better for complex backgrounds
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=True
        )
        
        # GMG - good for stationary camera
        # Note: GMG might not be available in all OpenCV versions
        try:
            self.bg_subtractor_gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
        except:
            self.bg_subtractor_gmg = None
    
    def setup_detection_parameters(self):
        """Configure detection parameters"""
        self.min_contour_area = 300  # Reduced for better small person detection
        self.max_contour_area = 50000  # Prevent very large false positives
        self.movement_threshold = 0.015  # Motion ratio threshold
        self.position_threshold = 3.0  # Minimum position change in pixels
        self.idle_frames_threshold = 45  # Frames before considering idle (1.5 seconds at 30fps)
        self.min_detection_confidence = 0.4  # Lower YOLO confidence for better recall
        
    def setup_tracking(self):
        """Initialize tracking variables"""
        self.tracked_objects = {}
        self.next_id = 0
        self.max_disappeared = 45  # Increased for better tracking continuity
        self.tracking_history_length = 30
        
    def load_better_yolo_model(self):
        """Load a better YOLO model for improved detection"""
        # Try different models in order of preference
        model_options = [
            "yolo11x.pt",    # Largest, most accurate
            "yolo11l.pt",    # Large, good balance
            "yolo11m.pt",    # Medium, faster
            "yolo11s.pt",    # Small, fast
            "yolo11n.pt"     # Nano, fastest
        ]
        
        for model_name in model_options:
            try:
                print(f"Attempting to load {model_name}...")
                model = YOLO(model_name)
                print(f"Successfully loaded {model_name}")
                return model
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        raise Exception("Could not load any YOLO model")
    
    def detect_persons_enhanced(self, frame, yolo_model):
        """Enhanced person detection with multiple methods"""
        person_boxes = []
        
        # YOLO detection with lower confidence threshold
        results = yolo_model(frame, verbose=False, conf=self.min_detection_confidence)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    
                    # Filter by size (remove very small or very large detections)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    if (area > self.min_contour_area and 
                        area < self.max_contour_area and
                        height > width * 0.8):  # Person aspect ratio check
                        person_boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
        
        return person_boxes
    
    def enhance_foreground_mask(self, mask):
        """Apply morphological operations to clean up the mask"""
        # Remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill holes
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Final smoothing
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def calculate_enhanced_iou(self, box1, box2):
        """Enhanced IoU calculation with aspect ratio consideration"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        # Consider center distance as well
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Normalize distance by box size
        avg_size = np.sqrt((area1 + area2) / 2)
        normalized_distance = distance / avg_size if avg_size > 0 else 1
        
        # Combine IoU with distance factor
        distance_factor = max(0, 1 - normalized_distance)
        enhanced_iou = iou * 0.7 + distance_factor * 0.3
        
        return enhanced_iou
    
    def track_objects_enhanced(self, current_boxes):
        """Enhanced object tracking with better matching"""
        if not self.tracked_objects:
            # Initialize tracking
            for i, box in enumerate(current_boxes):
                self.tracked_objects[self.next_id] = {
                    'box': box[:4],
                    'confidence': box[4] if len(box) > 4 else 1.0,
                    'disappeared': 0,
                    'positions': deque(maxlen=self.tracking_history_length),
                    'motion_history': deque(maxlen=20),
                    'last_motion_frame': 0,
                    'creation_frame': 0,
                    'stable_detections': 1
                }
                self.next_id += 1
            return list(self.tracked_objects.keys())
        
        # Enhanced matching with Hungarian algorithm simulation
        object_ids = list(self.tracked_objects.keys())
        matches = []
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(object_ids), len(current_boxes)))
        
        for i, obj_id in enumerate(object_ids):
            for j, box in enumerate(current_boxes):
                similarity = self.calculate_enhanced_iou(
                    self.tracked_objects[obj_id]['box'], box[:4]
                )
                similarity_matrix[i, j] = similarity
        
        # Simple matching based on highest similarity
        matched_pairs = []
        used_objects = set()
        used_boxes = set()
        
        while True:
            max_sim = 0
            best_match = None
            
            for i, obj_id in enumerate(object_ids):
                if i in used_objects:
                    continue
                for j, box in enumerate(current_boxes):
                    if j in used_boxes:
                        continue
                    if similarity_matrix[i, j] > max_sim and similarity_matrix[i, j] > 0.2:
                        max_sim = similarity_matrix[i, j]
                        best_match = (i, j, obj_id)
            
            if best_match is None:
                break
            
            i, j, obj_id = best_match
            matched_pairs.append((obj_id, j))
            used_objects.add(i)
            used_boxes.add(j)
        
        # Update matched objects
        active_ids = []
        
        for obj_id, box_idx in matched_pairs:
            box = current_boxes[box_idx]
            self.tracked_objects[obj_id]['box'] = box[:4]
            self.tracked_objects[obj_id]['confidence'] = box[4] if len(box) > 4 else 1.0
            self.tracked_objects[obj_id]['disappeared'] = 0
            self.tracked_objects[obj_id]['stable_detections'] += 1
            active_ids.append(obj_id)
        
        # Handle unmatched tracked objects
        for i, obj_id in enumerate(object_ids):
            if i not in used_objects:
                self.tracked_objects[obj_id]['disappeared'] += 1
        
        # Remove objects that have disappeared for too long
        to_remove = [obj_id for obj_id, obj_data in self.tracked_objects.items() 
                    if obj_data['disappeared'] > self.max_disappeared]
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
        
        # Add new objects for unmatched boxes
        for j, box in enumerate(current_boxes):
            if j not in used_boxes:
                self.tracked_objects[self.next_id] = {
                    'box': box[:4],
                    'confidence': box[4] if len(box) > 4 else 1.0,
                    'disappeared': 0,
                    'positions': deque(maxlen=self.tracking_history_length),
                    'motion_history': deque(maxlen=20),
                    'last_motion_frame': 0,
                    'creation_frame': 0,
                    'stable_detections': 1
                }
                active_ids.append(self.next_id)
                self.next_id += 1
        
        return active_ids
    
    def analyze_movement_multi_method(self, frame, frame_count):
        """Multi-method movement analysis"""
        # Get foreground masks from different methods
        fg_mask_mog2 = self.bg_subtractor_mog2.apply(frame)
        fg_mask_knn = self.bg_subtractor_knn.apply(frame)
        
        # Combine masks for more robust detection
        combined_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        if self.bg_subtractor_gmg is not None:
            try:
                fg_mask_gmg = self.bg_subtractor_gmg.apply(frame)
                combined_mask = cv2.bitwise_or(combined_mask, fg_mask_gmg)
            except:
                pass
        
        # Enhance the combined mask
        enhanced_mask = self.enhance_foreground_mask(combined_mask)
        
        movement_results = {}
        
        for person_id, person_data in self.tracked_objects.items():
            if person_data['disappeared'] > 0:
                continue
            
            # Only analyze stable detections
            if person_data['stable_detections'] < 3:
                movement_results[person_id] = {
                    'state': 'initializing',
                    'motion_ratio': 0,
                    'position_change': 0,
                    'frames_since_motion': 0,
                    'confidence': person_data['confidence']
                }
                continue
            
            box = person_data['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid ROI
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract ROI from enhanced mask
            roi_mask = enhanced_mask[y1:y2, x1:x2]
            
            if roi_mask.size == 0:
                continue
            
            # Calculate motion metrics
            motion_pixels = cv2.countNonZero(roi_mask)
            roi_area = (x2 - x1) * (y2 - y1)
            motion_ratio = motion_pixels / roi_area if roi_area > 0 else 0
            
            # Store data
            person_data['motion_history'].append(motion_ratio)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            person_data['positions'].append([center_x, center_y])
            
            # Analyze movement with multiple criteria
            if len(person_data['motion_history']) > 5:
                # Motion-based analysis
                recent_motion = np.mean(list(person_data['motion_history'])[-5:])
                motion_variance = np.var(list(person_data['motion_history'])[-10:])
                
                # Position-based analysis
                pos_change = 0
                if len(person_data['positions']) > 1:
                    pos_change = np.linalg.norm(
                        np.array(person_data['positions'][-1]) - 
                        np.array(person_data['positions'][-2])
                    )
                
                # Velocity analysis over longer period
                velocity = 0
                if len(person_data['positions']) > 10:
                    recent_positions = list(person_data['positions'])[-10:]
                    velocities = []
                    for i in range(1, len(recent_positions)):
                        vel = np.linalg.norm(
                            np.array(recent_positions[i]) - np.array(recent_positions[i-1])
                        )
                        velocities.append(vel)
                    velocity = np.mean(velocities)
                
                # Multi-criteria decision
                is_moving = False
                
                # Criterion 1: Motion in background subtraction
                if recent_motion > self.movement_threshold:
                    is_moving = True
                    person_data['last_motion_frame'] = frame_count
                
                # Criterion 2: Position change
                if pos_change > self.position_threshold:
                    is_moving = True
                    person_data['last_motion_frame'] = frame_count
                
                # Criterion 3: Consistent velocity
                if velocity > 1.5:
                    is_moving = True
                    person_data['last_motion_frame'] = frame_count
                
                # Criterion 4: Motion variance (indicates activity)
                if motion_variance > 0.001:
                    is_moving = True
                    person_data['last_motion_frame'] = frame_count
                
                # Final state determination
                frames_since_motion = frame_count - person_data['last_motion_frame']
                if frames_since_motion >= self.idle_frames_threshold:
                    final_state = 'idle'
                else:
                    final_state = 'moving'
                
                movement_results[person_id] = {
                    'state': final_state,
                    'motion_ratio': recent_motion,
                    'position_change': pos_change,
                    'velocity': velocity,
                    'frames_since_motion': frames_since_motion,
                    'confidence': person_data['confidence']
                }
            else:
                movement_results[person_id] = {
                    'state': 'analyzing',
                    'motion_ratio': motion_ratio,
                    'position_change': 0,
                    'velocity': 0,
                    'frames_since_motion': 0,
                    'confidence': person_data['confidence']
                }
        
        return movement_results, enhanced_mask
    
    def process_video_enhanced(self, video_path, output_path=None):
        """Enhanced video processing with better models and methods"""
        print("Loading enhanced YOLO model...")
        yolo_model = self.load_better_yolo_model()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print("Starting enhanced video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhanced person detection
            person_boxes = self.detect_persons_enhanced(frame, yolo_model)
            
            # Enhanced tracking
            active_ids = self.track_objects_enhanced(person_boxes)
            
            # Multi-method movement analysis
            movement_results, enhanced_mask = self.analyze_movement_multi_method(
                frame, frame_count
            )
            
            # Enhanced visualization
            display_frame = frame.copy()
            
            for person_id in active_ids:
                if person_id in movement_results:
                    box = self.tracked_objects[person_id]['box']
                    x1, y1, x2, y2 = map(int, box)
                    
                    result = movement_results[person_id]
                    state = result['state']
                    confidence = result['confidence']
                    
                    # Enhanced color coding
                    if state == 'moving':
                        color = (0, 255, 0)  # Green
                        thickness = 3
                    elif state == 'idle':
                        color = (0, 0, 255)  # Red
                        thickness = 3
                    elif state == 'analyzing':
                        color = (0, 255, 255)  # Yellow
                        thickness = 2
                    else:  # initializing
                        color = (128, 128, 128)  # Gray
                        thickness = 1
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Enhanced text display
                    text = f"ID:{person_id} {state.upper()}"
                    text2 = f"Conf:{confidence:.2f} M:{result['motion_ratio']:.3f}"
                    if 'velocity' in result:
                        text3 = f"Vel:{result['velocity']:.1f} Idle:{result['frames_since_motion']}"
                        cv2.putText(display_frame, text3, (x1, y2+35), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    cv2.putText(display_frame, text, (x1, y1-25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display_frame, text2, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw trajectory
                    if person_id in self.tracked_objects and len(self.tracked_objects[person_id]['positions']) > 1:
                        positions = list(self.tracked_objects[person_id]['positions'])
                        for i in range(1, min(len(positions), 10)):
                            pt1 = tuple(map(int, positions[-(i+1)]))
                            pt2 = tuple(map(int, positions[-i]))
                            alpha = (10-i)/10
                            trail_color = tuple(int(c * alpha) for c in color)
                            cv2.line(display_frame, pt1, pt2, trail_color, 2)
            
            # Display statistics
            moving_count = sum(1 for r in movement_results.values() if r['state'] == 'moving')
            idle_count = sum(1 for r in movement_results.values() if r['state'] == 'idle')
            
            stats_text = f"Frame: {frame_count} | Total: {len(active_ids)} | Moving: {moving_count} | Idle: {idle_count}"
            cv2.putText(display_frame, stats_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show enhanced mask alongside main frame
            mask_display = cv2.cvtColor(enhanced_mask, cv2.COLOR_GRAY2BGR)
            
            # Resize for display if needed
            h, w = display_frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                new_w, new_h = int(w * scale), int(h * scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                mask_display = cv2.resize(mask_display, (new_w, new_h))
            
            combined = np.hstack([display_frame, mask_display])
            cv2.imshow('Enhanced Movement Detection', combined)
            
            if output_path:
                out.write(display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Enhanced debug output
            if frame_count % 60 == 0:
                print(f"Frame {frame_count}: {len(active_ids)} persons tracked "
                      f"({moving_count} moving, {idle_count} idle)")
                for pid, result in movement_results.items():
                    if result['state'] in ['moving', 'idle']:
                        print(f"  Person {pid}: {result['state']} "
                              f"(motion: {result['motion_ratio']:.3f}, "
                              f"vel: {result.get('velocity', 0):.1f}, "
                              f"conf: {result['confidence']:.2f})")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Processed {frame_count} frames.")

# Usage
if __name__ == "__main__":
    # Create enhanced detector
    detector = EnhancedMovementDetector()
    
    # Process video with enhanced methods
    detector.process_video_enhanced(
        "assets/input_videos/clip1.mp4",
        output_path="enhanced_movement_detection.mp4"
    )