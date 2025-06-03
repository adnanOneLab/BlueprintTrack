from datetime import datetime
from shapely.geometry import Polygon

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history, confidence, timestamp}
        self.store_entry_threshold = 0.5
        self.max_frames_missing = 10  # Reduced for faster cleanup
        self.iou_threshold = 0.25  # Slightly reduced for better matching
        self.min_confidence = 0.60
        self.max_movement = 150  # Reduced to prevent large jumps
        self.velocity_smoothing = 0.5
        self.last_positions = {}  # Store last positions for velocity calculation
        self.track_history = {}  # Store recent positions for each track
        self.max_history = 3  # Reduced for less memory usage
        self.face_detection_history = {}  # Store recent face detections for each track
        self.max_face_history = 2
        self.last_store = {}  # Track last store for each person
        self.frame_counter = 0  # Add frame counter for better tracking

        # Adjust thresholds for CCTV footage (lower frame rate, lower resolution)
        self.movement_threshold = 5.0  # Increased from 3.0 to 5.0 pixels for CCTV
        self.movement_history = {}  # Store recent movement states
        self.movement_history_size = 2  # Reduced to 2 frames for CCTV (lower frame rate)
        self.velocity_threshold = 2.0  # Increased to 2.0 for CCTV footage
        self.velocity_smoothing = 0.5  # Reduced smoothing for more responsive movement detection
        
        # Add debugging flags
        self.debug_mode = True
        self.debug_movement = True  # Enable movement debugging
    
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
        """Update movement status with improved logic for CCTV footage"""
        # Calculate movement distance
        movement_distance = self.calculate_movement_distance(old_bbox, new_bbox)
        velocity_magnitude = self.calculate_velocity_magnitude(velocity)
        
        # For CCTV, we need to account for larger movements between frames
        # due to lower frame rate
        is_moving_distance = movement_distance > self.movement_threshold
        is_moving_velocity = velocity_magnitude > self.velocity_threshold
        
        # For CCTV, we want to be more responsive to movement
        # since frames are further apart
        current_moving = is_moving_distance or is_moving_velocity
        
        # Initialize movement history if needed
        if person_id not in self.movement_history:
            self.movement_history[person_id] = []
        
        # Add current movement state to history
        self.movement_history[person_id].append(current_moving)
        
        # Keep only recent history (shorter for CCTV due to lower frame rate)
        if len(self.movement_history[person_id]) > self.movement_history_size:
            self.movement_history[person_id].pop(0)
        
        # For CCTV, we want to be more responsive to movement changes
        # since frames are further apart
        moving_frames = sum(self.movement_history[person_id])
        total_frames = len(self.movement_history[person_id])
        
        # Person is moving if at least one frame shows movement
        # This is more appropriate for CCTV's lower frame rate
        person['is_moving'] = moving_frames > 0
        
        # Debug information
        if self.debug_movement:
            print(f"Person {person_id}: Distance={movement_distance:.1f}, "
                  f"Velocity={velocity_magnitude:.1f}, "
                  f"Moving={person['is_moving']} ({moving_frames}/{total_frames})")
    
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
            self.movement_history.pop(person_id, None)
        
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
            
            # Initialize missing fields
            if 'history' not in person:
                person['history'] = []
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
                self.movement_history[person_id] = [False]
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