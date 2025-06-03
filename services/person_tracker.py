from datetime import datetime
from shapely.geometry import Polygon
import numpy as np
from ultralytics import YOLO
import cv2

class PersonTracker:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history, confidence, timestamp, is_moving, position_history}
        self.store_entry_threshold = 0.5
        self.max_frames_missing = 10
        self.iou_threshold = 0.25
        self.min_confidence = 0.35  # Reduced from 0.60 to 0.35 to detect more people
        self.max_movement = 150
        self.velocity_smoothing = 0.5
        self.last_positions = {}
        self.track_history = {}
        self.max_history = 10
        self.face_detection_history = {}
        self.max_face_history = 2
        self.last_store = {}
        self.frame_counter = 0

        # YOLO class indices
        self.PERSON_CLASS = 0  # COCO dataset class index for person
        
        # Processing control from test.py
        self.frame_skip = 2  # Reduced from 5 to 2 for more frequent updates
        self.last_results = {
            'faces': [], 
            'persons': [],
            'objects': {}
        }
        
        # Drastically reduced movement thresholds for better sensitivity
        self.motion_threshold = 1.0  # Reduced from 3.0 to 1.0 pixels
        self.position_history_size = 3  # Reduced from 5 to 3 for faster response
        self.idle_threshold = 15  # Increased from 10 to 15 frames to be more stable
        
        # Movement detection thresholds - much more sensitive now
        self.min_movement_threshold = 0.5  # Reduced from 2.0 to 0.5
        self.max_movement_threshold = 100.0  # Increased from 50.0 to 100.0
        self.velocity_threshold = 0.3  # Reduced from 1.0 to 0.3
        self.movement_history_size = 2  # Reduced from 3 to 2 for faster response
        
        # Reduced state persistence for quicker state changes
        self.movement_persistence = 2  # Reduced from 3 to 2
        self.idle_persistence = 3  # Reduced from 5 to 3
        
        # Movement state tracking
        self.movement_state = {}
        
        # Visualization settings from test.py
        self.label_colors = {
            'moving': (255, 0, 255),  # Purple
            'idle': (255, 165, 0),    # Orange
            'person': (255, 0, 0),    # Blue
            'store': (0, 255, 0)      # Green
        }
        
        # Add debugging flags
        self.debug_mode = True
        self.debug_movement = True
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_velocity(self, old_bbox, new_bbox):
        """Calculate velocity between two bounding boxes"""
        old_center_x = old_bbox[0] + old_bbox[2] / 2
        old_center_y = old_bbox[1] + old_bbox[3] / 2
        new_center_x = new_bbox[0] + new_bbox[2] / 2
        new_center_y = new_bbox[1] + new_bbox[3] / 2
        
        return (new_center_x - old_center_x, new_center_y - old_center_y)
    
    def calculate_velocity_magnitude(self, velocity):
        """Calculate velocity magnitude from velocity vector"""
        return (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
    
    def predict_next_position(self, bbox, velocity):
        """Predict next position based on current position and velocity"""
        x, y, w, h = bbox
        dx, dy = velocity
        return (x + dx, y + dy, w, h)
    
    def calculate_movement(self, bbox1, bbox2):
        """Calculate the center point movement between two bounding boxes"""
        x1 = bbox1[0] + bbox1[2] / 2
        y1 = bbox1[1] + bbox1[3] / 2
        x2 = bbox2[0] + bbox2[2] / 2
        y2 = bbox2[1] + bbox2[3] / 2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def calculate_movement_distance(self, bbox1, bbox2):
        """Calculate the center point distance between two bounding boxes"""
        x1_center = bbox1[0] + bbox1[2] / 2
        y1_center = bbox1[1] + bbox1[3] / 2
        x2_center = bbox2[0] + bbox2[2] / 2
        y2_center = bbox2[1] + bbox2[3] / 2
        
        distance = ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5
        return distance
    
    def update_movement_status(self, person_id, person, old_bbox, new_bbox, velocity):
        """Update movement status with much more sensitive detection"""
        # Calculate center points
        curr_center = ((new_bbox[0] + new_bbox[2]) // 2, (new_bbox[1] + new_bbox[3]) // 2)
        prev_center = ((old_bbox[0] + old_bbox[2]) // 2, (old_bbox[1] + old_bbox[3]) // 2)
        
        # Calculate immediate movement
        movement_distance = self.calculate_movement_distance(old_bbox, new_bbox)
        velocity_magnitude = self.calculate_velocity_magnitude(velocity)
        
        # Initialize movement state if not exists
        if person_id not in self.movement_state:
            self.movement_state[person_id] = {
                'moving_frames': 0,
                'idle_frames': 0,
                'last_movement_time': 0,
                'movement_history': [],
                'position_history': [],
                'last_state': 'unknown',
                'state_persistence': 0,
                'consecutive_movements': 0  # Track consecutive movement frames
            }
        
        # Update position history
        self.movement_state[person_id]['position_history'].append(curr_center)
        if len(self.movement_state[person_id]['position_history']) > self.position_history_size:
            self.movement_state[person_id]['position_history'].pop(0)
        
        # Update movement history
        self.movement_state[person_id]['movement_history'].append(movement_distance)
        if len(self.movement_state[person_id]['movement_history']) > self.movement_history_size:
            self.movement_state[person_id]['movement_history'].pop(0)
        
        # Calculate various movement metrics
        recent_movements = self.movement_state[person_id]['movement_history']
        avg_movement = sum(recent_movements) / len(recent_movements) if recent_movements else 0
        
        # Calculate position-based movement
        position_history = self.movement_state[person_id]['position_history']
        position_movement = 0
        if len(position_history) > 1:
            total_movement = 0
            for i in range(1, len(position_history)):
                p1 = position_history[i-1]
                p2 = position_history[i]
                total_movement += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            position_movement = total_movement / (len(position_history) - 1)
        
        # Determine movement state using more sensitive criteria
        is_moving = False
        movement_reasons = []  # Track why movement was detected
        
        # Check immediate movement with very low threshold
        if movement_distance > self.min_movement_threshold:
            is_moving = True
            movement_reasons.append(f"dist={movement_distance:.1f}")
        
        # Check velocity with very low threshold
        if velocity_magnitude > self.velocity_threshold:
            is_moving = True
            movement_reasons.append(f"vel={velocity_magnitude:.1f}")
        
        # Check average movement
        if avg_movement > self.motion_threshold:
            is_moving = True
            movement_reasons.append(f"avg={avg_movement:.1f}")
        
        # Check position-based movement
        if position_movement > self.motion_threshold:
            is_moving = True
            movement_reasons.append(f"pos={position_movement:.1f}")
        
        # Update consecutive movements counter
        if is_moving:
            self.movement_state[person_id]['consecutive_movements'] += 1
        else:
            self.movement_state[person_id]['consecutive_movements'] = 0
        
        # Update movement state with persistence
        current_state = self.movement_state[person_id]['last_state']
        state_persistence = self.movement_state[person_id]['state_persistence']
        
        # More aggressive movement detection
        if is_moving or self.movement_state[person_id]['consecutive_movements'] > 0:
            if current_state != 'moving':
                if state_persistence <= 0:  # Only change state if persistence is exhausted
                    self.movement_state[person_id]['last_state'] = 'moving'
                    self.movement_state[person_id]['state_persistence'] = self.movement_persistence
                else:
                    self.movement_state[person_id]['state_persistence'] -= 1
            else:
                self.movement_state[person_id]['state_persistence'] = self.movement_persistence
        else:
            if current_state != 'idle':
                if state_persistence <= 0:  # Only change state if persistence is exhausted
                    self.movement_state[person_id]['last_state'] = 'idle'
                    self.movement_state[person_id]['state_persistence'] = self.idle_persistence
                else:
                    self.movement_state[person_id]['state_persistence'] -= 1
            else:
                self.movement_state[person_id]['state_persistence'] = self.idle_persistence
        
        # Update person's movement status
        person['is_moving'] = self.movement_state[person_id]['last_state'] == 'moving'
        person['is_idle'] = self.movement_state[person_id]['last_state'] == 'idle'
        person['movement_state'] = self.movement_state[person_id]['last_state']
        person['avg_movement'] = avg_movement
        person['position_movement'] = position_movement
        person['last_movement_distance'] = movement_distance
        person['velocity_magnitude'] = velocity_magnitude
        
        # Debug information with movement reasons
        if self.debug_movement:
            reasons_str = ", ".join(movement_reasons) if movement_reasons else "no movement"
            print(f"Person {person_id}: "
                  f"Dist={movement_distance:.1f}, "
                  f"Vel={velocity_magnitude:.1f}, "
                  f"Avg={avg_movement:.1f}, "
                  f"Pos={position_movement:.1f}, "
                  f"State={person['movement_state']}, "
                  f"Persistence={self.movement_state[person_id]['state_persistence']}, "
                  f"Consecutive={self.movement_state[person_id]['consecutive_movements']}, "
                  f"Reasons=[{reasons_str}]")
    
    def is_person_in_store(self, person_bbox, store_polygon):
        """Check if a person is inside a store using polygon intersection"""
        try:
            # Create person bounding box polygon
            x, y, w, h = person_bbox
            person_polygon = Polygon([
                (x, y), (x + w, y), (x + w, y + h), (x, y + h)
            ])
            
            # Create store polygon
            store_polygon = Polygon(store_polygon)
            
            # Calculate intersection
            if person_polygon.intersects(store_polygon):
                intersection = person_polygon.intersection(store_polygon)
                # If more than threshold of person is in store, count as entry
                return (intersection.area / person_polygon.area) > self.store_entry_threshold
            
            return False
        except Exception as e:
            if self.debug_mode:
                print(f"Error checking store entry: {str(e)}")
            return False
    
    def cleanup_old_tracks(self, current_frame):
        """Clean up old tracks that haven't been seen for a while"""
        tracks_to_remove = []
        
        for person_id, person in self.tracked_people.items():
            frames_missing = current_frame - person['last_seen']
            if frames_missing >= self.max_frames_missing:
                tracks_to_remove.append(person_id)
                if self.debug_mode:
                    print(f"Removing track {person_id} - missing for {frames_missing} frames")
        
        # Remove old tracks from all dictionaries
        for person_id in tracks_to_remove:
            self.tracked_people.pop(person_id, None)
            self.last_positions.pop(person_id, None)
            self.track_history.pop(person_id, None)
            self.face_detection_history.pop(person_id, None)
            self.last_store.pop(person_id, None)
        
        if self.debug_mode and tracks_to_remove:
            print(f"Cleaned up {len(tracks_to_remove)} old tracks. Active tracks: {len(self.tracked_people)}")
    
    def find_best_match(self, detection, unmatched_tracks):
        """Find the best matching track for a detection"""
        best_score = 0
        best_track_id = None
        
        for person_id in unmatched_tracks:
            person = self.tracked_people[person_id]
            
            # Get current or predicted bbox
            current_bbox = person['bbox']
            if person_id in self.last_positions:
                velocity = self.last_positions[person_id].get('velocity', (0, 0))
                predicted_bbox = self.predict_next_position(current_bbox, velocity)
            else:
                predicted_bbox = current_bbox
            
            # Calculate IOU with both current and predicted positions
            current_iou = self.calculate_iou(current_bbox, detection['bbox'])
            predicted_iou = self.calculate_iou(predicted_bbox, detection['bbox'])
            iou = max(current_iou, predicted_iou)
            
            # Calculate movement score
            movement = self.calculate_movement(current_bbox, detection['bbox'])
            movement_score = max(0, 1 - (movement / self.max_movement))
            
            # Calculate size similarity score
            old_area = current_bbox[2] * current_bbox[3]
            new_area = detection['bbox'][2] * detection['bbox'][3]
            size_ratio = min(old_area, new_area) / max(old_area, new_area) if max(old_area, new_area) > 0 else 0
            
            # Combined score with balanced weights
            score = (0.5 * iou) + (0.3 * movement_score) + (0.2 * size_ratio)
            
            if score > best_score and iou >= 0.1:  # Minimum IOU threshold
                best_score = score
                best_track_id = person_id
        
        return best_track_id, best_score
    
    def update_store_status(self, person_id, person, stores, current_time, frame_number):
        """Update store status for a person"""
        current_store = None
        
        for store_id, store in stores.items():
            if "video_polygon" in store and len(store["video_polygon"]) > 2:
                if self.is_person_in_store(person['bbox'], store["video_polygon"]):
                    current_store = store_id
                    # Check if this is a new store entry
                    if current_store != self.last_store.get(person_id):
                        entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        if self.debug_mode:
                            print(f"Person {person_id} entered {store.get('name', 'Unknown')} at {entry_time}")
                        
                        # Ensure history exists
                        if 'history' not in person:
                            person['history'] = []
                        
                        # Add to history
                        person['history'].append({
                            'store_name': store.get('name', 'Unknown'),
                            'entry_time': entry_time,
                            'frame': frame_number
                        })
                    break
        
        # Update store tracking
        person['current_store'] = current_store
        self.last_store[person_id] = current_store
    
    def update(self, detected_people, stores, frame_number, face_detections=None):
        """Update tracked people with new detections"""
        self.frame_counter = frame_number
        current_time = datetime.now()
        
        # Filter detections by confidence
        detected_people = [p for p in detected_people if p.get('confidence', 0) >= self.min_confidence]
        
        if self.debug_mode:
            print(f"\nFrame {frame_number}: Processing {len(detected_people)} detections, {len(self.tracked_people)} active tracks")
        
        # Clean up old tracks first
        self.cleanup_old_tracks(frame_number)
        
        # Initialize tracking data for existing tracks
        for person_id in self.tracked_people:
            person = self.tracked_people[person_id]
            
            # Initialize missing fields with improved defaults from test.py
            if 'history' not in person:
                person['history'] = []
            if 'position_history' not in person:
                person['position_history'] = []
            if person_id not in self.track_history:
                self.track_history[person_id] = []
            if person_id not in self.face_detection_history:
                self.face_detection_history[person_id] = []
            if person_id not in self.last_store:
                self.last_store[person_id] = None
        
        # Match detections to existing tracks
        matched_detections = set()
        unmatched_tracks = set(self.tracked_people.keys())
        
        # Process each detection and find best match
        for i, detection in enumerate(detected_people):
            if not unmatched_tracks:
                break
                
            best_track_id, best_score = self.find_best_match(detection, unmatched_tracks)
            
            if best_track_id is not None and best_score > self.iou_threshold:
                # Update existing track
                person = self.tracked_people[best_track_id]
                old_bbox = person['bbox']
                new_bbox = detection['bbox']
                
                # Calculate and smooth velocity
                velocity = self.calculate_velocity(old_bbox, new_bbox)
                if best_track_id in self.last_positions:
                    old_velocity = self.last_positions[best_track_id].get('velocity', (0, 0))
                    velocity = (
                        self.velocity_smoothing * old_velocity[0] + (1 - self.velocity_smoothing) * velocity[0],
                        self.velocity_smoothing * old_velocity[1] + (1 - self.velocity_smoothing) * velocity[1]
                    )

                # IMPROVED: Update movement status with better logic
                self.update_movement_status(best_track_id, person, old_bbox, new_bbox, velocity)
                
                # Update person data
                person['bbox'] = new_bbox
                person['confidence'] = detection['confidence']
                person['last_seen'] = frame_number
                person['timestamp'] = detection.get('timestamp', current_time.timestamp())

                # Store movement distance for debugging
                person['last_movement_distance'] = self.calculate_movement_distance(old_bbox, new_bbox)
                person['velocity_magnitude'] = self.calculate_velocity_magnitude(velocity)
                
                # Update velocity and position history
                self.last_positions[best_track_id] = {
                    'velocity': velocity,
                    'last_update': frame_number
                }
                
                # Update track history
                self.track_history[best_track_id].append(new_bbox)
                if len(self.track_history[best_track_id]) > self.max_history:
                    self.track_history[best_track_id].pop(0)
                
                # Update store status
                self.update_store_status(best_track_id, person, stores, current_time, frame_number)
                
                # Mark as matched
                matched_detections.add(i)
                unmatched_tracks.remove(best_track_id)
                
                if self.debug_mode:
                    print(f"  Matched detection {i} to track {best_track_id} (score: {best_score:.3f})")
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detected_people):
            if i not in matched_detections:
                person_id = self.next_id
                self.next_id += 1
                
                # Initialize new track
                self.tracked_people[person_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'last_seen': frame_number,
                    'current_store': None,
                    'history': [],
                    'timestamp': detection.get('timestamp', current_time.timestamp()),
                    'is_moving': False,  # New tracks start as not moving
                    'last_movement_distance': 0.0,
                    'velocity_magnitude': 0.0
                }
                
                # Initialize tracking data
                self.last_positions[person_id] = {
                    'velocity': (0, 0),
                    'last_update': frame_number
                }

                self.track_history[person_id] = [detection['bbox']]
                self.face_detection_history[person_id] = []
                self.last_store[person_id] = None
                
                # Update store status for new track
                self.update_store_status(person_id, self.tracked_people[person_id], stores, current_time, frame_number)
                
                if self.debug_mode:
                    print(f"  Created new track {person_id} for unmatched detection {i}")
        
        # Update last_seen for all remaining unmatched tracks (they weren't updated this frame)
        for person_id in unmatched_tracks:
            # Don't update last_seen - let them age out naturally
            pass
        
        if self.debug_mode:
            print(f"Frame {frame_number} complete: {len(self.tracked_people)} active tracks")
        
        return self.tracked_people

    def analyze_frame(self, frame):
        """Analyze a single frame with YOLO (from test.py)"""
        if frame is None or frame.size == 0:
            return {'faces': [], 'persons': [], 'objects': {}}
        
        # Run YOLO detection with optimized settings
        results = self.model(frame, 
                           verbose=False,
                           conf=self.min_confidence,  # Set confidence threshold
                           iou=0.45,  # IOU threshold for NMS
                           max_det=50,  # Maximum number of detections
                           classes=[self.PERSON_CLASS]  # Only detect persons
                           )[0]
        
        # Process detections
        detections = {
            'faces': [],
            'persons': [],
            'objects': {}
        }
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()  # Convert to numpy array
            
            # Convert to (left, top, right, bottom) format
            left, top, right, bottom = map(int, xyxy)
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            if right <= left or bottom <= top:
                continue
            
            # Store detection based on class
            if cls == self.PERSON_CLASS:
                # Convert to (x, y, w, h) format for our tracker
                x, y, w, h = left, top, right - left, bottom - top
                detections['persons'].append({
                    'bbox': (x, y, w, h),
                    'confidence': conf
                })
                # Also add to faces for tracking
                detections['faces'].append({
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'name': 'Unknown'
                })
            else:
                # Store other objects
                class_name = results.names[cls]
                if class_name not in detections['objects']:
                    detections['objects'][class_name] = []
                detections['objects'][class_name].append({
                    'bbox': (x, y, w, h),
                    'confidence': conf
                })
        
        if self.debug_mode:
            print(f"YOLO detected {len(detections['persons'])} people")
        
        return detections

    def process_frame(self, frame, stores):
        """Process a single frame with YOLO detection and tracking"""
        if frame is None or frame.size == 0:
            return self.tracked_people
            
        # Skip frames based on frame_skip setting
        if self.frame_counter % self.frame_skip != 0:
            self.frame_counter += 1
            return self.tracked_people
            
        # Analyze frame with YOLO
        detections = self.analyze_frame(frame)
        
        # Update tracking with detected persons
        self.update(detections['persons'], stores, self.frame_counter, detections['faces'])
        
        self.frame_counter += 1
        return self.tracked_people

    def draw_detections(self, frame):
        """Draw detections and tracking information on frame with improved movement visualization"""
        if frame is None or frame.size == 0:
            return frame
            
        frame_with_detections = frame.copy()
        
        # Draw tracked people
        for person_id, person in self.tracked_people.items():
            x, y, w, h = person['bbox']
            
            # Get movement state and determine color
            movement_state = person.get('movement_state', 'unknown')
            if movement_state == 'moving':
                color = self.label_colors['moving']  # Purple
                status = "Moving"
            elif movement_state == 'idle':
                color = self.label_colors['idle']    # Orange
                status = "Idle"
            else:
                color = self.label_colors['person']  # Blue
                status = "Unknown"
            
            # Draw bounding box with thicker line for better visibility
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), color, 2)
            
            # Draw movement status with background for better readability
            label = f"ID:{person_id} {status}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_with_detections, 
                         (x, y - label_h - 10), 
                         (x + label_w, y), 
                         color, -1)  # Filled rectangle
            cv2.putText(frame_with_detections, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
            
            # Draw movement metrics if debug mode is on
            if self.debug_mode:
                metrics = [
                    f"Dist: {person.get('last_movement_distance', 0):.1f}",
                    f"Vel: {person.get('velocity_magnitude', 0):.1f}",
                    f"Avg: {person.get('avg_movement', 0):.1f}"
                ]
                for i, metric in enumerate(metrics):
                    y_pos = y + h + 20 + (i * 20)
                    cv2.putText(frame_with_detections, metric, (x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw store information if available
            if person.get('current_store'):
                store_label = f"Store: {person['current_store']}"
                cv2.putText(frame_with_detections, store_label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.label_colors['store'], 2)
        
        # Draw frame counter with background
        counter_text = f"Frame: {self.frame_counter}"
        (counter_w, counter_h), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame_with_detections, 
                     (10, 10), 
                     (20 + counter_w, 40), 
                     (0, 0, 0), -1)  # Black background
        cv2.putText(frame_with_detections, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame_with_detections