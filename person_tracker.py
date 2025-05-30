from datetime import datetime
from shapely.geometry import Polygon

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history, last_store}
        self.store_entry_threshold = 0.5
        self.max_frames_missing = 30
        self.iou_threshold = 0.3
        self.min_confidence = 0.75  # Match AWS confidence threshold
        self.max_movement = 150
        self.velocity_smoothing = 0.7
        self.last_positions = {}  # Store last positions for velocity calculation
        self.notified_entries = set()  # Track which store entries we've already notified
        self.notified_exits = set()  # Track which store exits we've already notified
    
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
    
    def update(self, detected_people, stores, frame_number):
        """Update tracked people with new detections"""
        current_time = datetime.now()
        
        # Filter detections by confidence
        detected_people = [p for p in detected_people if p['confidence'] >= self.min_confidence]
        
        # Update existing tracks with predictions
        for person_id in list(self.tracked_people.keys()):
            person = self.tracked_people[person_id]
            if person_id in self.last_positions:
                velocity = self.last_positions[person_id].get('velocity', (0, 0))
                predicted_bbox = self.predict_next_position(person['bbox'], velocity)
                person['predicted_bbox'] = predicted_bbox
            person['last_seen'] = frame_number
            
            # Check for store exit if person was in a store
            if person['current_store'] is not None:
                # Check if person is still in any store
                in_store = False
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(person['bbox'], store["video_polygon"]):
                            in_store = True
                            if person['current_store'] != store_id:
                                # Person moved to a different store
                                old_store = person['current_store']
                                person['current_store'] = store_id
                                # Log store change
                                entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                person['history'].append({
                                    'store_id': store_id,
                                    'store_name': store.get('name', 'Unknown'),
                                    'entry_time': entry_time,
                                    'frame': frame_number,
                                    'type': 'entry'
                                })
                                print(f"Person {person_id} moved from {stores[old_store].get('name', 'Unknown')} to {store.get('name', 'Unknown')} at {entry_time}")
                            break
                
                if not in_store:
                    # Person has exited the store
                    exit_key = f"{person_id}_{person['current_store']}"
                    if exit_key not in self.notified_exits:
                        exit_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        person['history'].append({
                            'store_id': person['current_store'],
                            'store_name': stores[person['current_store']].get('name', 'Unknown'),
                            'exit_time': exit_time,
                            'frame': frame_number,
                            'type': 'exit'
                        })
                        print(f"Person {person_id} exited {stores[person['current_store']].get('name', 'Unknown')} at {exit_time}")
                        self.notified_exits.add(exit_key)
                    person['current_store'] = None
        
        # Match new detections to existing tracks
        matched_detections = set()
        for person_id, person in self.tracked_people.items():
            best_score = 0
            best_detection = None
            
            for i, detection in enumerate(detected_people):
                if i in matched_detections:
                    continue
                
                bbox_to_compare = person.get('predicted_bbox', person['bbox'])
                iou = self.calculate_iou(bbox_to_compare, detection['bbox'])
                movement = self.calculate_movement(bbox_to_compare, detection['bbox'])
                movement_score = 1 - (movement / self.max_movement) if movement < self.max_movement else 0
                score = (0.7 * iou) + (0.3 * movement_score)
                
                if score > self.iou_threshold and score > best_score:
                    best_score = score
                    best_detection = (i, detection)
            
            if best_detection:
                idx, detection = best_detection
                matched_detections.add(idx)
                
                # Calculate velocity for next prediction
                old_bbox = person['bbox']
                new_bbox = detection['bbox']
                velocity = self.calculate_velocity(old_bbox, new_bbox)
                
                # Smooth velocity
                if person_id in self.last_positions:
                    old_velocity = self.last_positions[person_id].get('velocity', (0, 0))
                    velocity = (
                        self.velocity_smoothing * old_velocity[0] + (1 - self.velocity_smoothing) * velocity[0],
                        self.velocity_smoothing * old_velocity[1] + (1 - self.velocity_smoothing) * velocity[1]
                    )
                
                # Update person data
                person['bbox'] = new_bbox
                person['confidence'] = detection['confidence']
                person['timestamp'] = detection.get('timestamp', current_time.timestamp())
                
                # Store velocity for next prediction
                self.last_positions[person_id] = {
                    'velocity': velocity,
                    'last_update': frame_number
                }
                
                # Check store entry
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(new_bbox, store["video_polygon"]):
                            if person['current_store'] != store_id:
                                # Only notify if we haven't notified this person-store combination before
                                entry_key = f"{person_id}_{store_id}"
                                if entry_key not in self.notified_entries:
                                    entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                    person['history'].append({
                                        'store_id': store_id,
                                        'store_name': store.get('name', 'Unknown'),
                                        'entry_time': entry_time,
                                        'frame': frame_number,
                                        'type': 'entry'
                                    })
                                    print(f"Person {person_id} entered {store.get('name', 'Unknown')} at {entry_time}")
                                    self.notified_entries.add(entry_key)
                            person['current_store'] = store_id
                            break
        
        # Add new tracks for unmatched detections
        for i, detection in enumerate(detected_people):
            if i not in matched_detections:
                person_id = self.next_id
                self.next_id += 1
                
                # Initialize with zero velocity
                self.last_positions[person_id] = {
                    'velocity': (0, 0),
                    'last_update': frame_number
                }
                
                # Check initial store
                current_store = None
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(detection['bbox'], store["video_polygon"]):
                            current_store = store_id
                            # Add to notified entries for new detections in stores
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
                
                if current_store:
                    entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    self.tracked_people[person_id]['history'].append({
                        'store_id': current_store,
                        'store_name': stores[current_store].get('name', 'Unknown'),
                        'entry_time': entry_time,
                        'frame': frame_number,
                        'type': 'entry'
                    })
        
        # Clean up old tracks and positions
        current_frame = frame_number
        self.tracked_people = {
            pid: person for pid, person in self.tracked_people.items()
            if current_frame - person['last_seen'] < self.max_frames_missing
        }
        self.last_positions = {
            pid: pos for pid, pos in self.last_positions.items()
            if current_frame - pos['last_update'] < self.max_frames_missing
        }
        
        return self.tracked_people