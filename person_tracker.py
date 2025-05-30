from datetime import datetime
from shapely.geometry import Polygon

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_people = {}  # id -> {bbox, last_seen, current_store, history}
        self.store_entry_threshold = 0.5  # How much of the person needs to be in store to count as entry
        self.max_frames_missing = 30  # How many frames a person can be missing before removing
        self.iou_threshold = 0.3  # IOU threshold for matching detections to tracked people
    
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
        
        # Update existing tracks
        for person_id in list(self.tracked_people.keys()):
            person = self.tracked_people[person_id]
            person['last_seen'] = frame_number
            person['current_store'] = None  # Reset current store, will update if still in one
        
        # Match new detections to existing tracks
        matched_detections = set()
        for person_id, person in self.tracked_people.items():
            best_iou = 0
            best_detection = None
            
            for i, detection in enumerate(detected_people):
                if i in matched_detections:
                    continue
                
                iou = self.calculate_iou(person['bbox'], detection['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_detection = (i, detection)
            
            if best_detection:
                idx, detection = best_detection
                matched_detections.add(idx)
                person['bbox'] = detection['bbox']
                person['confidence'] = detection['confidence']
                
                # Check store entry
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(detection['bbox'], store["video_polygon"]):
                            if person['current_store'] != store_id:
                                # Person entered a new store
                                entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                person['history'].append({
                                    'store_id': store_id,
                                    'store_name': store.get('name', 'Unknown'),
                                    'entry_time': entry_time,
                                    'frame': frame_number
                                })
                                # Update the last store entry for notification
                                if hasattr(self, 'parent') and isinstance(self.parent, CCTVPreview):
                                    self.parent.last_store_entry = {
                                        'person_id': person_id,
                                        'store_name': store.get('name', 'Unknown'),
                                        'entry_time': entry_time
                                    }
                                    self.parent.store_entry_display_time = self.parent.notification_duration
                                print(f"Person {person_id} entered {store.get('name', 'Unknown')} at {entry_time}")
                            person['current_store'] = store_id
                            break
        
        # Add new tracks for unmatched detections
        for i, detection in enumerate(detected_people):
            if i not in matched_detections:
                person_id = self.next_id
                self.next_id += 1
                
                # Check initial store
                current_store = None
                for store_id, store in stores.items():
                    if "video_polygon" in store and len(store["video_polygon"]) > 2:
                        if self.is_person_in_store(detection['bbox'], store["video_polygon"]):
                            current_store = store_id
                            break
                
                self.tracked_people[person_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'last_seen': frame_number,
                    'current_store': current_store,
                    'history': []
                }
                
                if current_store:
                    entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    self.tracked_people[person_id]['history'].append({
                        'store_id': current_store,
                        'store_name': stores[current_store].get('name', 'Unknown'),
                        'entry_time': entry_time,
                        'frame': frame_number
                    })
                    print(f"Person {person_id} entered {stores[current_store].get('name', 'Unknown')} at {entry_time}")
        
        # Remove old tracks
        self.tracked_people = {
            pid: person for pid, person in self.tracked_people.items()
            if frame_number - person['last_seen'] < self.max_frames_missing
        }
        
        return self.tracked_people