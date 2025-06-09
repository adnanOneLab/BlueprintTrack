import cv2
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path
import face_recognition
from sklearn.cluster import DBSCAN
from collections import defaultdict
import pickle

class PersonRecognitionSystem:
    """Enhanced person recognition using saved body images and face encodings"""
    
    def __init__(self, body_images_dir="exports/body_images/"):
        self.body_images_dir = body_images_dir
        self.face_encodings_file = os.path.join(body_images_dir, "face_encodings.pkl")
        self.person_database_file = os.path.join(body_images_dir, "person_database.json")
        
        # Face recognition settings
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_recognition_tolerance = 0.6
        
        # Person database
        self.person_database = {}  # person_name -> {encodings, images, first_seen, last_seen}
        
        # Clustering settings for automatic person grouping
        self.clustering_eps = 0.4
        self.clustering_min_samples = 2
        
        # Load existing data
        self.load_person_database()
        self.load_face_encodings()
        
        # Processing flags
        self.auto_clustering_enabled = True
        self.face_detection_confidence = 0.8
        
    def load_person_database(self):
        """Load person database from JSON file"""
        try:
            if os.path.exists(self.person_database_file):
                with open(self.person_database_file, 'r') as f:
                    self.person_database = json.load(f)
                print(f"Loaded person database with {len(self.person_database)} people")
            else:
                self.person_database = {}
        except Exception as e:
            print(f"Error loading person database: {e}")
            self.person_database = {}
    
    def save_person_database(self):
        """Save person database to JSON file"""
        try:
            with open(self.person_database_file, 'w') as f:
                json.dump(self.person_database, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving person database: {e}")
    
    def load_face_encodings(self):
        """Load face encodings from pickle file"""
        try:
            if os.path.exists(self.face_encodings_file):
                with open(self.face_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_encodings)} face encodings")
            else:
                self.known_face_encodings = []
                self.known_face_names = []
        except Exception as e:
            print(f"Error loading face encodings: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_face_encodings(self):
        """Save face encodings to pickle file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving face encodings: {e}")
    
    def extract_face_encoding(self, image_path):
        """Extract face encoding from an image"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                return None
            
            # Get face encoding (use the first face found)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if face_encodings:
                return face_encodings[0]
            
        except Exception as e:
            print(f"Error extracting face encoding from {image_path}: {e}")
        
        return None
    
    def process_saved_images(self):
        """Process all saved body images to extract face encodings"""
        if not os.path.exists(self.body_images_dir):
            print(f"Body images directory not found: {self.body_images_dir}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(self.body_images_dir).glob(f"*{ext}"))
        
        print(f"Processing {len(image_files)} saved images...")
        
        # Process each image
        processed_count = 0
        encodings_extracted = 0
        
        for image_path in image_files:
            # Skip enhanced images to avoid duplicates
            if "enhanced" in str(image_path):
                continue
                
            try:
                # Extract face encoding
                encoding = self.extract_face_encoding(str(image_path))
                
                if encoding is not None:
                    # Extract timestamp from filename
                    filename = image_path.stem
                    timestamp_str = filename.split('_', 1)[-1] if '_' in filename else filename
                    
                    # Add to known encodings
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(f"Person_{timestamp_str}")
                    
                    encodings_extracted += 1
                    
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} images, extracted {encodings_extracted} encodings...")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        print(f"Processing complete: {encodings_extracted} face encodings extracted from {processed_count} images")
        
        # Perform automatic clustering
        if self.auto_clustering_enabled and encodings_extracted > 0:
            self.cluster_faces()
        
        # Save encodings
        self.save_face_encodings()
    
    def cluster_faces(self):
        """Automatically cluster faces to group the same person"""
        if len(self.known_face_encodings) < 2:
            return
        
        print("Performing face clustering to group same persons...")
        
        # Convert to numpy array for clustering
        encodings_array = np.array(self.known_face_encodings)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.clustering_eps, 
                          min_samples=self.clustering_min_samples, 
                          metric='euclidean').fit(encodings_array)
        
        # Group faces by cluster
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(clustering.labels_):
            clusters[cluster_id].append(i)
        
        # Update names based on clusters
        cluster_counter = 1
        updated_names = self.known_face_names.copy()
        
        for cluster_id, face_indices in clusters.items():
            if cluster_id == -1:  # Noise/outliers
                continue
                
            if len(face_indices) >= self.clustering_min_samples:
                # This is a valid cluster - assign same person name
                person_name = f"Person_{cluster_counter:03d}"
                
                for face_idx in face_indices:
                    updated_names[face_idx] = person_name
                
                cluster_counter += 1
        
        self.known_face_names = updated_names
        
        # Update person database
        self.update_person_database_from_clustering()
        
        print(f"Clustering complete: Found {cluster_counter-1} unique persons")
    
    def update_person_database_from_clustering(self):
        """Update person database based on clustering results"""
        person_groups = defaultdict(list)
        
        # Group by person name
        for i, name in enumerate(self.known_face_names):
            person_groups[name].append(i)
        
        # Update database
        for person_name, encoding_indices in person_groups.items():
            if person_name not in self.person_database:
                self.person_database[person_name] = {
                    'encoding_count': len(encoding_indices),
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'total_detections': len(encoding_indices)
                }
            else:
                self.person_database[person_name]['encoding_count'] = len(encoding_indices)
                self.person_database[person_name]['last_seen'] = datetime.now().isoformat()
                self.person_database[person_name]['total_detections'] += len(encoding_indices)
        
        self.save_person_database()
    
    def recognize_face_in_frame(self, frame, face_bbox):
        """Recognize a face in the current frame using known encodings"""
        try:
            x, y, w, h = face_bbox
            
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Convert BGR to RGB (face_recognition expects RGB)
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_locations = [(0, w, h, 0)]  # top, right, bottom, left
            face_encodings = face_recognition.face_encodings(face_rgb, face_locations)
            
            if not face_encodings:
                return None, 0.0
            
            face_encoding = face_encodings[0]
            
            # Compare with known faces
            if self.known_face_encodings:
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                if min_distance <= self.face_recognition_tolerance:
                    confidence = 1.0 - min_distance
                    return self.known_face_names[min_distance_idx], confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, 0.0
    
    def add_new_person_encoding(self, frame, face_bbox, person_name=None):
        """Add a new person encoding from current frame"""
        try:
            if person_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                person_name = f"Person_{timestamp}"
            
            # Extract and add encoding
            x, y, w, h = face_bbox
            face_image = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            face_locations = [(0, w, h, 0)]
            face_encodings = face_recognition.face_encodings(face_rgb, face_locations)
            
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(person_name)
                
                # Update database
                if person_name not in self.person_database:
                    self.person_database[person_name] = {
                        'encoding_count': 1,
                        'first_seen': datetime.now().isoformat(),
                        'last_seen': datetime.now().isoformat(),
                        'total_detections': 1
                    }
                
                # Save data
                self.save_face_encodings()
                self.save_person_database()
                
                return True
        
        except Exception as e:
            print(f"Error adding new person encoding: {e}")
        
        return False
    
    def get_person_stats(self):
        """Get statistics about recognized persons"""
        stats = {
            'total_persons': len(set(self.known_face_names)),
            'total_encodings': len(self.known_face_encodings),
            'person_details': {}
        }
        
        # Count encodings per person
        person_counts = defaultdict(int)
        for name in self.known_face_names:
            person_counts[name] += 1
        
        for person, count in person_counts.items():
            stats['person_details'][person] = {
                'encoding_count': count,
                'database_info': self.person_database.get(person, {})
            }
        
        return stats


# Modified PersonTracker class to integrate with recognition system
class EnhancedPersonTracker:
    """Enhanced PersonTracker with face recognition capabilities"""
    
    def __init__(self, body_images_dir="exports/body_images/"):
        # Initialize base tracker (copy all the initialization from your original PersonTracker)
        # ... (include all the original initialization code here)
        
        # Add recognition system
        self.recognition_system = PersonRecognitionSystem(body_images_dir)
        
        # Recognition settings
        self.enable_face_recognition = True
        self.recognition_confidence_threshold = 0.7
        self.person_names = {}  # person_id -> recognized_name
        
        # Process existing images on startup
        self.recognition_system.process_saved_images()
    
    def update_with_recognition(self, detected_people, stores, frame_number, face_detections=None, current_frame=None):
        """Enhanced update method that includes face recognition"""
        # First, do the normal tracking update
        tracked_people = self.update(detected_people, stores, frame_number, face_detections)
        
        # Then, perform face recognition if enabled and frame is provided
        if self.enable_face_recognition and current_frame is not None and face_detections:
            self.perform_face_recognition(current_frame, face_detections, tracked_people)
        
        return tracked_people
    
    def perform_face_recognition(self, frame, face_detections, tracked_people):
        """Perform face recognition on detected faces"""
        for person_id, person in tracked_people.items():
            person_bbox = person['bbox']
            
            # Find overlapping face detections
            for face_detection in face_detections:
                face_bbox = face_detection.get('bbox')
                if face_bbox and self.boxes_overlap(person_bbox, face_bbox):
                    # Perform recognition
                    recognized_name, confidence = self.recognition_system.recognize_face_in_frame(
                        frame, face_bbox
                    )
                    
                    if recognized_name and confidence >= self.recognition_confidence_threshold:
                        # Update person name
                        self.person_names[person_id] = {
                            'name': recognized_name,
                            'confidence': confidence,
                            'last_recognition': datetime.now()
                        }
                        
                        if self.debug_mode:
                            print(f"Recognized Person {person_id} as {recognized_name} (confidence: {confidence:.2f})")
                    
                    break  # Only process one face per person
    
    def boxes_overlap(self, bbox1, bbox2, overlap_threshold=0.3):
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if right <= left or bottom <= top:
            return False
        
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate overlap ratio
        overlap_ratio = intersection / min(area1, area2)
        return overlap_ratio >= overlap_threshold
    
    def draw_detections_with_names(self, frame):
        """Enhanced drawing method that shows recognized names"""
        if frame is None or frame.size == 0:
            return frame
            
        frame_with_detections = frame.copy()
        
        # Draw tracked people with names
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
            
            # Draw bounding box
            cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label with name if recognized
            if person_id in self.person_names:
                person_info = self.person_names[person_id]
                label = f"{person_info['name']} ({person_info['confidence']:.2f})"
                # Use green color for recognized persons
                color = (0, 255, 0)
            else:
                label = f"ID:{person_id} {status}"
            
            # Draw label with background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_with_detections, 
                         (x, y - label_h - 10), 
                         (x + label_w, y), 
                         color, -1)
            cv2.putText(frame_with_detections, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw additional info
            if self.debug_mode:
                info_lines = [
                    f"Dist: {person.get('last_movement_distance', 0):.1f}",
                    f"Vel: {person.get('velocity_magnitude', 0):.1f}",
                    status
                ]
                
                for i, info in enumerate(info_lines):
                    y_pos = y + h + 20 + (i * 20)
                    cv2.putText(frame_with_detections, info, (x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw recognition stats
        stats = self.recognition_system.get_person_stats()
        stats_text = f"Known Persons: {stats['total_persons']} | Encodings: {stats['total_encodings']}"
        cv2.putText(frame_with_detections, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_with_detections