from datetime import datetime
from shapely.geometry import Polygon

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history, last_store, face_detections, body_images}
        self.store_entry_threshold = 0.5
        self.max_frames_missing = 15  # Reduced from 30 to remove ghost boxes faster
        self.iou_threshold = 0.3  # Increased from 0.2 for better matching
        self.min_confidence = 0.60  # Match AWS confidence threshold
        self.max_movement = 200  # Reduced from 300 to prevent large jumps
        self.velocity_smoothing = 0.8  # Reduced from 0.95 to make tracking more responsive
        self.last_positions = {}  # Store last positions for velocity calculation
        self.notified_entries = set()  # Track which store entries we've already notified
        self.notified_exits = set()  # Track which store exits we've already notified
        self.track_history = {}  # Store recent positions for each track
        self.max_history = 5  # Reduced from 10 to prevent ghost trails
        self.active_tracks = set()  # Keep track of currently active track IDs
        self.face_detection_history = {}  # Store recent face detections for each track
        self.max_face_history = 3  # Reduced from 5 to keep only recent face detections
    
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
            print(f"Error checking store entry: {str(e)}")
            return False
    
    def update(self, detected_people, stores, frame_number, face_detections=None):
        """Update tracked people with new detections and face detections"""
        current_time = datetime.now()
        
        # Filter detections by confidence
        detected_people = [p for p in detected_people if p['confidence'] >= self.min_confidence]
        if face_detections:
            face_detections = [f for f in face_detections if f['confidence'] >= self.min_confidence]
        
        # Update track history for existing tracks
        for person_id in self.tracked_people:
            if person_id not in self.track_history:
                self.track_history[person_id] = []
            self.track_history[person_id].append(self.tracked_people[person_id]['bbox'])
            if len(self.track_history[person_id]) > self.max_history:
                self.track_history[person_id].pop(0)
            
            # Initialize face detection history if not exists
            if person_id not in self.face_detection_history:
                self.face_detection_history[person_id] = []
        
        # Update existing tracks with predictions
        for person_id in list(self.tracked_people.keys()):
            person = self.tracked_people[person_id]
            if person_id in self.last_positions:
                velocity = self.last_positions[person_id].get('velocity', (0, 0))
                predicted_bbox = self.predict_next_position(person['bbox'], velocity)
                person['predicted_bbox'] = predicted_bbox
            person['last_seen'] = frame_number
            self.active_tracks.add(person_id)
        
        # Match new detections to existing tracks
        matched_detections = set()
        unmatched_tracks = set(self.tracked_people.keys())
        
        # First pass: Try to match with high confidence using predicted positions
        for person_id in list(unmatched_tracks):
            person = self.tracked_people[person_id]
            best_score = 0
            best_detection = None
            best_face = None
            
            # Try to match with face detections first if available
            if face_detections:
                for face in face_detections:
                    face_bbox = face['bbox']
                    bbox_to_compare = person.get('predicted_bbox', person['bbox'])
                    
                    # Calculate IOU between face and predicted person bbox
                    iou = self.calculate_iou(bbox_to_compare, face_bbox)
                    
                    # Calculate movement score
                    movement = self.calculate_movement(bbox_to_compare, face_bbox)
                    movement_score = 1 - (movement / self.max_movement) if movement < self.max_movement else 0
                    
                    # Combined score with adjusted weights
                    score = (0.6 * iou) + (0.4 * movement_score)  # Increased weight for IOU
                    
                    if score > best_score:
                        best_score = score
                        best_detection = None  # Clear person detection
                        best_face = face
            
            # If no good face match, try person detection
            if best_score < self.iou_threshold:
                for i, detection in enumerate(detected_people):
                    if i in matched_detections:
                        continue
                    
                    # Try matching with predicted position first
                    bbox_to_compare = person.get('predicted_bbox', person['bbox'])
                    iou = self.calculate_iou(bbox_to_compare, detection['bbox'])
                    
                    # Skip if IOU is too low
                    if iou < 0.15:  # Increased from 0.1 for better matching
                        continue
                    
                    # Calculate movement score
                    movement = self.calculate_movement(bbox_to_compare, detection['bbox'])
                    movement_score = 1 - (movement / self.max_movement) if movement < self.max_movement else 0
                    
                    # Calculate history score
                    history_score = 0
                    if person_id in self.track_history and len(self.track_history[person_id]) > 0:
                        history_ious = [self.calculate_iou(hist_bbox, detection['bbox']) 
                                      for hist_bbox in self.track_history[person_id]]
                        history_score = max(history_ious) if history_ious else 0
                    
                    # Combined score with adjusted weights
                    score = (0.5 * iou) + (0.3 * movement_score) + (0.2 * history_score)
                    
                    if score > best_score:
                        best_score = score
                        best_detection = (i, detection)
                        best_face = None  # Clear face detection
            
            if best_score > self.iou_threshold:
                if best_face:
                    # Update with face detection
                    matched_detections.add(-1)  # Use -1 to indicate face detection was matched
                    unmatched_tracks.remove(person_id)
                    
                    # Update person data
                    old_bbox = person['bbox']
                    new_bbox = best_face['body_bbox']  # Use body bbox from face detection
                    
                    # Calculate and smooth velocity
                    velocity = self.calculate_velocity(old_bbox, new_bbox)
                    if person_id in self.last_positions:
                        old_velocity = self.last_positions[person_id].get('velocity', (0, 0))
                        velocity = (
                            self.velocity_smoothing * old_velocity[0] + (1 - self.velocity_smoothing) * velocity[0],
                            self.velocity_smoothing * old_velocity[1] + (1 - self.velocity_smoothing) * velocity[1]
                        )
                    
                    person['bbox'] = new_bbox
                    person['confidence'] = best_face['confidence']
                    person['timestamp'] = best_face['timestamp']
                    
                    # Store face detection
                    if 'face_detections' not in person:
                        person['face_detections'] = []
                    person['face_detections'].append({
                        'bbox': best_face['bbox'],
                        'confidence': best_face['confidence'],
                        'timestamp': best_face['timestamp'],
                        'image_path': best_face.get('image_path')
                    })
                    if len(person['face_detections']) > self.max_face_history:
                        person['face_detections'].pop(0)
                    
                    # Update face detection history
                    self.face_detection_history[person_id].append(best_face)
                    if len(self.face_detection_history[person_id]) > self.max_face_history:
                        self.face_detection_history[person_id].pop(0)
                    
                elif best_detection:
                    # Update with person detection
                    idx, detection = best_detection
                    matched_detections.add(idx)
                    unmatched_tracks.remove(person_id)
                    
                    # Update person data
                    old_bbox = person['bbox']
                    new_bbox = detection['bbox']
                    
                    # Calculate and smooth velocity
                    velocity = self.calculate_velocity(old_bbox, new_bbox)
                    if person_id in self.last_positions:
                        old_velocity = self.last_positions[person_id].get('velocity', (0, 0))
                        velocity = (
                            self.velocity_smoothing * old_velocity[0] + (1 - self.velocity_smoothing) * velocity[0],
                            self.velocity_smoothing * old_velocity[1] + (1 - self.velocity_smoothing) * velocity[1]
                        )
                    
                    person['bbox'] = new_bbox
                    person['confidence'] = detection['confidence']
                    person['timestamp'] = detection.get('timestamp', current_time.timestamp())
                
                # Update velocity and position history (common for both face and person detection)
                self.last_positions[person_id] = {
                    'velocity': velocity,
                    'last_update': frame_number
                }
                
                # Update track history
                if person_id not in self.track_history:
                    self.track_history[person_id] = []
                self.track_history[person_id].append(person['bbox'])
                if len(self.track_history[person_id]) > self.max_history:
                    self.track_history[person_id].pop(0)
        
        # Clean up old tracks that haven't been seen for a while
        current_frame = frame_number
        self.tracked_people = {
            pid: person for pid, person in self.tracked_people.items()
            if current_frame - person['last_seen'] < self.max_frames_missing
        }
        self.last_positions = {
            pid: pos for pid, pos in self.last_positions.items()
            if current_frame - pos['last_update'] < self.max_frames_missing
        }
        self.track_history = {
            pid: history for pid, history in self.track_history.items()
            if pid in self.tracked_people
        }
        self.face_detection_history = {
            pid: history for pid, history in self.face_detection_history.items()
            if pid in self.tracked_people
        }
        
        # Add new tracks for unmatched detections
        for i, detection in enumerate(detected_people):
            if i not in matched_detections:
                # Find the next available ID that hasn't been used recently
                while self.next_id in self.active_tracks:
                    self.next_id += 1
                
                person_id = self.next_id
                self.next_id += 1
                self.active_tracks.add(person_id)
                
                # Initialize with zero velocity
                self.last_positions[person_id] = {
                    'velocity': (0, 0),
                    'last_update': frame_number
                }
                
                # Initialize track history
                self.track_history[person_id] = [detection['bbox']]
                
                # Check initial store
                current_store = None
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(detection['bbox'], store["video_polygon"]):
                            current_store = store_id
                            entry_key = f"{person_id}_{store_id}"
                            if entry_key not in self.notified_entries:
                                entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                print(f"Person {person_id} entered {store.get('name', 'Unknown')} at {entry_time}")
                                self.notified_entries.add(entry_key)
                            break
                
                self.tracked_people[person_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'last_seen': frame_number,
                    'current_store': current_store,
                    'history': [],
                    'timestamp': detection.get('timestamp', current_time.timestamp())
                }
        
        # Clean up active tracks set
        self.active_tracks = set(self.tracked_people.keys())
        
        return self.tracked_people