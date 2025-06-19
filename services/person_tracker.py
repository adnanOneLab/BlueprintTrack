from datetime import datetime
from shapely.geometry import Polygon
import numpy as np
from ultralytics import YOLO
import cv2
import torch

class PersonTracker:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolo11n.pt')  # Using YOLOv11 nano model
        self.location = "Unknown"  # Will be updated from stores mapping
        
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history, confidence, timestamp, is_moving, position_history}
        self.store_entry_threshold = 0.5
        self.max_frames_missing = 10
        self.iou_threshold = 0.15
        self.min_confidence = 0.25
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
        self.PERSON_CLASS = 0
        
        # Processing control
        self.frame_skip = 20
        self.last_results = {
            'faces': [], 
            'persons': [],
            'objects': {}
        }
        
        # Export mode optimization
        self.is_export_mode = False
        self.export_frame_skip = 3  # More aggressive skipping during export
        
        # IMPROVED: CCTV-optimized movement thresholds
        self.motion_threshold = 2.0  # Decreased further for CCTV's lower frame rate
        self.position_history_size = 4  # Reduced for faster response in CCTV
        self.idle_threshold = 15  # Reduced for CCTV's lower frame rate
        
        # IMPROVED: CCTV-optimized movement detection thresholds
        self.min_movement_threshold = 1.5  # Much lower for CCTV's lower frame rate
        self.max_movement_threshold = 200.0  # Increased to handle faster movements
        self.velocity_threshold = 1.0  # Lower for CCTV's lower frame rate
        self.movement_history_size = 3  # Reduced for faster response
        
        # IMPROVED: Faster state changes for CCTV
        self.movement_persistence = 2  # Reduced for faster response
        self.idle_persistence = 3  # Reduced for faster response
        
        # NEW: CCTV-specific parameters
        self.min_consecutive_movement_frames = 1  # Single frame movement detection
        self.jitter_threshold = 0.8  # Lower to detect smaller movements
        self.confidence_movement_factor = 0.5  # More lenient confidence requirements
        
        # NEW: CCTV-specific movement detection
        self.movement_scale_factor = 1.5  # Scale movement based on person size
        self.min_person_height = 50  # Minimum height to consider for movement
        self.max_person_height = 400  # Maximum height to consider for movement
        
        # Movement state tracking
        self.movement_state = {}
        
        # Visualization settings - standardized color scheme
        self.label_colors = {
            'moving': (0, 165, 255),    # Orange for moving (BGR format)
            'idle': (255, 0, 0),        # Blue for idle (BGR format)
            'person': (255, 0, 0),      # Blue for person (BGR format)
            'store': (0, 255, 0),       # Green for store (BGR format)
            'face': (0, 0, 255)         # Red for face detection (BGR format)
        }
        
        # Debugging flags
        self.debug_mode = True
        self.debug_movement = True

    def set_export_mode(self, enabled):
        """Enable/disable export mode for performance optimization"""
        if enabled and not self.is_export_mode:
            # Entering export mode - save current state
            self.saved_tracking_state = self.save_tracking_state()
            self.is_export_mode = enabled
            # Reset frame counter for clean export state
            self.frame_counter = 0
            print("PersonTracker: Export mode enabled - using aggressive frame skipping")
        elif not enabled and self.is_export_mode:
            # Exiting export mode - restore previous state
            self.is_export_mode = enabled
            if hasattr(self, 'saved_tracking_state'):
                self.restore_tracking_state(self.saved_tracking_state)
                self.saved_tracking_state = None
            print("PersonTracker: Export mode disabled - restored previous tracking state")
        else:
            # No state change needed
            self.is_export_mode = enabled
            if enabled:
                print("PersonTracker: Export mode already enabled")
            else:
                print("PersonTracker: Export mode already disabled")

    def detect_camera_jitter(self, movement_history):
        """Detect if movements are likely due to camera jitter/noise"""
        if len(movement_history) < 3:
            return False
        
        # Check if all movements are very small
        small_movements = sum(1 for m in movement_history if m < self.jitter_threshold)
        jitter_ratio = small_movements / len(movement_history)
        
        # If most movements are tiny, likely camera jitter
        return jitter_ratio > 0.7
    
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
        """Calculate the center point distance between two bounding boxes with CCTV adjustments"""
        x1_center = bbox1[0] + bbox1[2] / 2
        y1_center = bbox1[1] + bbox1[3] / 2
        x2_center = bbox2[0] + bbox2[2] / 2
        y2_center = bbox2[1] + bbox2[3] / 2
        
        # Calculate base distance
        distance = ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5
        
        # Scale movement based on person size (larger in frame = more movement needed)
        person_height = max(bbox1[3], bbox2[3])
        if self.min_person_height <= person_height <= self.max_person_height:
            # Scale factor increases as person gets larger in frame
            scale = 1.0 + (person_height - self.min_person_height) / (self.max_person_height - self.min_person_height)
            distance = distance / (scale * self.movement_scale_factor)
        
        return distance
    
    def calculate_movement_stability(self, position_history):
        """Calculate how stable/consistent movement is over time"""
        if len(position_history) < 3:
            return 0.0
        
        # Calculate distances between consecutive positions
        distances = []
        for i in range(1, len(position_history)):
            p1 = position_history[i-1]
            p2 = position_history[i]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Calculate coefficient of variation (std/mean)
        mean_dist = np.mean(distances)
        if mean_dist < 0.1:  # Very small movements
            return 0.0
        
        std_dist = np.std(distances)
        cv = std_dist / mean_dist
        
        # Lower CV = more consistent movement
        # Higher CV = more erratic/jittery movement
        stability = max(0, 1.0 - cv)
        return stability
    
    def update_movement_status(self, person_id, person, old_bbox, new_bbox, velocity):
        """Simplified movement detection with consolidated thresholds"""
        # Calculate movement distance
        movement_distance = self.calculate_movement_distance(old_bbox, new_bbox)
        velocity_magnitude = self.calculate_velocity_magnitude(velocity)
        
        # Initialize movement state if not exists
        if person_id not in self.movement_state:
            self.movement_state[person_id] = {
                'moving_frames': 0,
                'idle_frames': 0,
                'last_state': 'idle',
                'state_persistence': self.idle_persistence
            }
        
        state = self.movement_state[person_id]
        
        # Simplified movement detection - only use distance and velocity
        person_height = new_bbox[3]
        is_valid_size = self.min_person_height <= person_height <= self.max_person_height
        
        # Determine if currently moving based on simple criteria
        is_currently_moving = (
            is_valid_size and
            movement_distance > self.min_movement_threshold and
            velocity_magnitude > self.velocity_threshold
        )
        
        # Update state with simplified persistence logic
        current_state = state['last_state']
        
        if is_currently_moving:
            if current_state != 'moving':
                if state['state_persistence'] <= 0:
                    state['last_state'] = 'moving'
                    state['state_persistence'] = self.movement_persistence
                    if self.debug_mode:
                        print(f"[{self.location}] Person {person_id}: State changed to MOVING")
                else:
                    state['state_persistence'] -= 1
            else:
                state['state_persistence'] = self.movement_persistence
        else:
            if current_state != 'idle':
                if state['state_persistence'] <= 0:
                    state['last_state'] = 'idle'
                    state['state_persistence'] = self.idle_persistence
                    if self.debug_mode:
                        print(f"[{self.location}] Person {person_id}: State changed to IDLE")
                else:
                    state['state_persistence'] -= 1
            else:
                state['state_persistence'] = self.idle_persistence
        
        # Update person's movement status
        person['is_moving'] = state['last_state'] == 'moving'
        person['is_idle'] = state['last_state'] == 'idle'
        person['movement_state'] = state['last_state']
        person['last_movement_distance'] = movement_distance
        person['velocity_magnitude'] = velocity_magnitude
        person['person_height'] = person_height
    
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
        
        # Remove old tracks from all dictionaries
        for person_id in tracks_to_remove:
            self.tracked_people.pop(person_id, None)
            self.last_positions.pop(person_id, None)
            self.track_history.pop(person_id, None)
            self.face_detection_history.pop(person_id, None)
            self.last_store.pop(person_id, None)
            self.movement_state.pop(person_id, None)  # Clean up movement state to prevent memory leaks
    
    def find_best_match(self, detection, unmatched_tracks):
        """IMPROVED: Enhanced matching with movement prediction and size consistency"""
        best_score = 0
        best_track_id = None
        
        for person_id in unmatched_tracks:
            person = self.tracked_people[person_id]
            current_bbox = person['bbox']
            
            # Enhanced prediction using movement state
            if person_id in self.last_positions:
                velocity = self.last_positions[person_id].get('velocity', (0, 0))
                # Scale prediction based on movement state
                movement_state = self.movement_state.get(person_id, {})
                if movement_state.get('last_state') == 'moving':
                    predicted_bbox = self.predict_next_position(current_bbox, velocity)
                else:
                    # For idle objects, use minimal prediction
                    scaled_velocity = (velocity[0] * 0.1, velocity[1] * 0.1)
                    predicted_bbox = self.predict_next_position(current_bbox, scaled_velocity)
            else:
                predicted_bbox = current_bbox
            
            # Calculate multiple IOU scores
            current_iou = self.calculate_iou(current_bbox, detection['bbox'])
            predicted_iou = self.calculate_iou(predicted_bbox, detection['bbox'])
            best_iou = max(current_iou, predicted_iou)
            
            # Calculate movement constraint
            movement = self.calculate_movement(current_bbox, detection['bbox'])
            
            # Penalize excessive movement for idle tracks
            movement_penalty = 1.0
            if person_id in self.movement_state:
                if self.movement_state[person_id].get('last_state') == 'idle' and movement > 20:
                    movement_penalty = 0.5  # Penalize large movements for idle objects
            
            movement_score = max(0, 1 - (movement / self.max_movement)) * movement_penalty
            
            # Enhanced size similarity with aspect ratio
            old_w, old_h = current_bbox[2], current_bbox[3]
            new_w, new_h = detection['bbox'][2], detection['bbox'][3]
            
            old_area = old_w * old_h
            new_area = new_w * new_h
            area_ratio = min(old_area, new_area) / max(old_area, new_area) if max(old_area, new_area) > 0 else 0
            
            old_aspect = old_w / old_h if old_h > 0 else 1
            new_aspect = new_w / new_h if new_h > 0 else 1
            aspect_ratio = min(old_aspect, new_aspect) / max(old_aspect, new_aspect) if max(old_aspect, new_aspect) > 0 else 0
            
            size_score = (area_ratio + aspect_ratio) / 2
            
            # Confidence consistency bonus
            old_conf = person.get('confidence', 0.5)
            new_conf = detection.get('confidence', 0.5)
            conf_consistency = 1 - abs(old_conf - new_conf)
            
            # Combined score with balanced weights
            score = (0.4 * best_iou +           # Primary: IOU
                    0.25 * movement_score +      # Movement constraint
                    0.2 * size_score +           # Size consistency
                    0.15 * conf_consistency)     # Confidence consistency
            
            # Minimum thresholds
            if best_iou >= 0.1 and score > best_score:
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
                            print(f"[{self.location}] Person {person_id} entered {store.get('name', 'Unknown')} at {entry_time}")
                        
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
    
    def validate_detection(self, detection):
        """Validate detection data before creating tracks"""
        if not isinstance(detection, dict):
            return False
        
        # Check required fields
        if 'bbox' not in detection or 'confidence' not in detection:
            return False
        
        # Validate bbox format and values
        bbox = detection['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        
        x, y, w, h = bbox
        if not all(isinstance(v, (int, float)) for v in bbox):
            return False
        
        # Validate bbox dimensions
        if w <= 0 or h <= 0:
            return False
        
        # Validate confidence
        confidence = detection['confidence']
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return False
        
        return True

    def create_new_track(self, detection, frame_number, current_time, stores):
        """Create a new track with validation"""
        if not self.validate_detection(detection):
            if self.debug_mode:
                print(f"Invalid detection data, skipping track creation: {detection}")
            return None
        
        person_id = self.next_id
        self.next_id += 1
        
        # Initialize new track with validated data
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
            print(f"Created new track {person_id} with bbox {detection['bbox']}")
        
        return person_id

    def update(self, detected_people, stores, frame_number, face_detections=None):
        """Update tracked people with new detections"""
        self.frame_counter = frame_number
        current_time = datetime.now()
        
        # Standardize store data structure handling
        if isinstance(stores, dict):
            # Handle both flat dict and nested dict structures
            if "stores" in stores and isinstance(stores["stores"], dict):
                # Nested structure: stores = {"stores": {...}, "location": "..."}
                actual_stores = stores["stores"]
                self.location = stores.get("location", "Unknown")
            else:
                # Flat structure: stores = {"store1": {...}, "store2": {...}}
                actual_stores = stores
                # Try to get location from first store or use default
                first_store = next(iter(actual_stores.values()), None) if actual_stores else None
                self.location = first_store.get("location", "Unknown") if first_store else "Unknown"
        else:
            # Fallback for invalid store data
            actual_stores = {}
            self.location = "Unknown"
        
        # Filter detections by confidence
        detected_people = [p for p in detected_people if p.get('confidence', 0) >= self.min_confidence]
        
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
                
                # Update store status using standardized stores
                self.update_store_status(best_track_id, person, actual_stores, current_time, frame_number)
                
                # Mark as matched
                matched_detections.add(i)
                unmatched_tracks.remove(best_track_id)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detected_people):
            if i not in matched_detections:
                person_id = self.create_new_track(detection, frame_number, current_time, actual_stores)
                # Track creation is handled inside create_new_track method
        
        # Update last_seen for all remaining unmatched tracks (they weren't updated this frame)
        for person_id in unmatched_tracks:
            # Don't update last_seen - let them age out naturally
            pass
        
        return self.tracked_people

    def analyze_frame(self, frame):
        """IMPROVED: Enhanced YOLO detection with better filtering"""
        if frame is None or frame.size == 0:
            return {'faces': [], 'persons': [], 'objects': {}}
        
        # Run YOLO detection with optimized settings for performance
        try:
            # Try to use GPU if available, fallback to CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            results = self.model(frame, 
                            verbose=False,
                            conf=self.min_confidence,
                            iou=0.4,      # Slightly higher to reduce duplicate detections
                            max_det=20,   # Reduced for better performance
                            classes=[self.PERSON_CLASS],
                            device=device,  # Use GPU if available
                            half=True if device == 'cuda' else False  # Use half precision on GPU
                            )[0]
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"GPU detection failed, falling back to CPU: {str(e)}")
            results = self.model(frame, 
                            verbose=False,
                            conf=self.min_confidence,
                            iou=0.4,
                            max_det=20,
                            classes=[self.PERSON_CLASS],
                            device='cpu',
                            half=False
                            )[0]
        
        detections = {
            'faces': [],
            'persons': [],
            'objects': {}
        }
        
        # Filter and process detections with enhanced criteria
        valid_detections = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Convert coordinates
            left, top, right, bottom = map(int, xyxy)
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            if right <= left or bottom <= top:
                continue
                
            # Calculate detection properties for filtering
            w, h = right - left, bottom - top
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Enhanced filtering criteria
            min_area = 400  # Minimum area for valid detection
            max_area = (width * height) * 0.8  # Maximum 80% of frame
            min_aspect = 0.2  # Minimum aspect ratio
            max_aspect = 5.0  # Maximum aspect ratio
            
            # Filter out invalid detections
            if (area < min_area or area > max_area or 
                aspect_ratio < min_aspect or aspect_ratio > max_aspect):
                continue
            
            # Store valid detection
            detection = {
                'bbox': (left, top, w, h),
                'confidence': conf,
                'area': area,
                'aspect_ratio': aspect_ratio
            }
            valid_detections.append(detection)
        
        # Sort by confidence and take best detections
        valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add to results
        for detection in valid_detections[:15]:  # Reduced limit for better performance
            detections['persons'].append({
                'bbox': detection['bbox'],
                'confidence': detection['confidence']
            })
            detections['faces'].append({
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'name': 'Unknown'
            })
        
        return detections

    def process_frame(self, frame, stores):
        """Process a single frame with YOLO detection and tracking"""
        if frame is None or frame.size == 0:
            return self.tracked_people
            
        # Use export mode frame skipping if in export mode
        current_frame_skip = self.export_frame_skip if self.is_export_mode else self.frame_skip
        
        # Skip frames based on current frame_skip setting
        if self.frame_counter % current_frame_skip != 0:
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

    def save_tracking_state(self):
        """Save current tracking state for persistence"""
        return {
            'tracked_people': self.tracked_people.copy(),
            'last_positions': self.last_positions.copy(),
            'track_history': {k: v.copy() for k, v in self.track_history.items()},
            'last_store': self.last_store.copy(),
            'movement_state': {k: v.copy() for k, v in self.movement_state.items()},
            'next_id': self.next_id,
            'frame_counter': self.frame_counter
        }
    
    def restore_tracking_state(self, state):
        """Restore tracking state from saved data"""
        if state is None:
            return
        
        try:
            self.tracked_people = state.get('tracked_people', {})
            self.last_positions = state.get('last_positions', {})
            self.track_history = state.get('track_history', {})
            self.last_store = state.get('last_store', {})
            self.movement_state = state.get('movement_state', {})
            self.next_id = state.get('next_id', 1)
            self.frame_counter = state.get('frame_counter', 0)
            
            if self.debug_mode:
                print(f"Restored tracking state with {len(self.tracked_people)} tracks")
        except Exception as e:
            print(f"Error restoring tracking state: {str(e)}")
            # Reset to clean state if restoration fails
            self._reset_tracking_state()
    
    def _reset_tracking_state(self):
        """Reset tracking state to clean initial state"""
        self.tracked_people = {}
        self.last_positions = {}
        self.track_history = {}
        self.face_detection_history = {}
        self.last_store = {}
        self.movement_state = {}
        self.next_id = 1
        self.frame_counter = 0