import cv2
import time
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import torch

class VideoAnalyzer:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        
        # Visualization settings
        self.label_colors = {
            'face': (0, 255, 0),      # Green
            'person': (255, 0, 0),    # Blue
            'object': (0, 0, 255),    # Red
            'idle': (255, 165, 0),    # Orange
            'moving': (255, 0, 255)   # Purple
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
        # Processing control
        self.frame_skip = 5  # Process every 5th frame to balance performance
        self.frame_counter = 0
        self.last_results = {
            'faces': [], 
            'persons': [],
            'objects': defaultdict(list)
        }
        self.tracked_faces = {}  # To maintain face tracking between frames
        self.tracking_counter = 0
        
        # Person movement tracking
        self.tracked_persons = {}  # To track persons
        self.person_tracking_counter = 0
        self.motion_threshold = 10  # Pixel distance threshold to determine movement

    def process_video(self, video_path):
        """Process a video file with YOLO"""
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Processing video: {video_path}")
        print(f"📏 Resolution: {width}x{height}")
        print(f"🎞️ FPS: {fps:.1f}, Total frames: {total_frames}")
        print("🛑 Press 'q' to stop processing early\n")

        # Create output file
        output_path = video_path.split("/")[1].split(".")[0] + '_yolo_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_time = time.time()
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip empty frames
            if frame is None or frame.size == 0:
                self.frame_counter += 1
                continue

            self.frame_counter += 1
            
            # Process every N frames
            if self.frame_counter % self.frame_skip == 0:
                processed_frames += 1
                self.last_results = self.analyze_frame(frame)
                if self.last_results['faces']:
                    self.update_tracked_faces(frame)
                # Update tracked persons (including those not identified by face detection)
                self.update_tracked_persons(frame)
            
            # Draw detections on frame
            frame_with_detections = self.draw_detections(frame.copy())
            
            # Display processing info
            cv2.putText(frame_with_detections, 
                       f"Frame: {self.frame_counter}/{total_frames}", 
                       (10, 30), self.font, 0.8, (255, 255, 255), 2)
            
            # Write frame to output
            out.write(frame_with_detections)
            
            # Display preview
            cv2.imshow("Video Analysis (Press Q to stop)", frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        processing_time = time.time() - start_time
        print(f"\n✅ Analysis complete!")
        print(f"⏱️ Processed {processed_frames} frames in {processing_time:.1f} seconds")
        print(f"💾 Output saved to: {output_path}")

    def analyze_frame(self, frame):
        """Analyze a single frame with YOLO"""
        if frame is None or frame.size == 0:
            return {'faces': [], 'persons': [], 'objects': defaultdict(list)}
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)[0]
        
        # Process detections
        detections = {
            'faces': [],
            'persons': [],
            'objects': defaultdict(list)
        }
        
        # YOLO class indices for persons and faces
        PERSON_CLASS = 0  # COCO dataset class index for person
        FACE_CLASS = 0    # We'll use person detections for face tracking
        
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
            if cls == PERSON_CLASS:
                if conf > 0.5:  # Confidence threshold for persons
                    detections['persons'].append({
                        'box': (left, top, right, bottom),
                        'confidence': conf
                    })
                    # Also add to faces for tracking
                    detections['faces'].append({
                        'box': (left, top, right, bottom),
                        'confidence': conf,
                        'name': 'Unknown'  # We'll use this for tracking
                    })
            else:
                # Store other objects
                class_name = results.names[cls]
                detections['objects'][class_name].append({
                    'box': (left, top, right, bottom),
                    'confidence': conf
                })
        
        return detections

    def update_tracked_faces(self, frame):
        """Update tracked faces with new detections"""
        height, width = frame.shape[:2]
        current_boxes = [det['box'] for det in self.last_results['faces']]
        
        # Update existing tracks or create new ones
        updated_tracks = {}
        for i, (box, det) in enumerate(zip(current_boxes, self.last_results['faces'])):
            matched = False
            for track_id, track_data in self.tracked_faces.items():
                prev_box = track_data['box']
                iou = self.calculate_iou(box, prev_box)
                if iou > 0.3:  # Threshold for same face
                    updated_tracks[track_id] = {
                        'box': box,
                        'name': track_data['name'],  # Keep existing name
                        'confidence': det['confidence'],
                        'frames_since_update': 0
                    }
                    matched = True
                    break
            
            if not matched:
                new_id = self.tracking_counter
                updated_tracks[new_id] = {
                    'box': box,
                    'name': f"Person_{new_id}",  # Assign new ID
                    'confidence': det['confidence'],
                    'frames_since_update': 0
                }
                self.tracking_counter += 1
        
        # Keep unmatched tracks for a few frames
        for track_id, track_data in self.tracked_faces.items():
            if track_id not in updated_tracks:
                track_data['frames_since_update'] += 1
                if track_data['frames_since_update'] < 5:
                    updated_tracks[track_id] = track_data
        
        self.tracked_faces = updated_tracks

    def update_tracked_persons(self, frame):
        """Track persons and detect if they're moving or idle"""
        height, width = frame.shape[:2]
        current_boxes = [det['box'] for det in self.last_results['persons']]
        
        # Update existing person tracks or create new ones
        updated_tracks = {}
        for box in current_boxes:
            matched = False
            for person_id, person_data in self.tracked_persons.items():
                prev_box = person_data['box']
                iou = self.calculate_iou(box, prev_box)
                if iou > 0.3:  # Threshold for same person
                    # Calculate center points to determine movement
                    curr_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                    prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
                    
                    # Store positions for movement history
                    position_history = person_data.get('position_history', [])
                    position_history.append(curr_center)
                    if len(position_history) > 10:
                        position_history = position_history[-10:]
                    
                    # Calculate average movement
                    avg_movement = 0
                    if len(position_history) > 1:
                        movements = []
                        for i in range(1, len(position_history)):
                            p1 = position_history[i-1]
                            p2 = position_history[i]
                            movements.append(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
                        avg_movement = sum(movements) / len(movements)
                    
                    is_moving = avg_movement > self.motion_threshold
                    
                    updated_tracks[person_id] = {
                        'box': box,
                        'frames_since_update': 0,
                        'position_history': position_history,
                        'is_moving': is_moving,
                        'avg_movement': avg_movement
                    }
                    matched = True
                    break
            
            if not matched:
                new_id = f"person_{self.person_tracking_counter}"
                curr_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                updated_tracks[new_id] = {
                    'box': box,
                    'frames_since_update': 0,
                    'position_history': [curr_center],
                    'is_moving': False,
                    'avg_movement': 0
                }
                self.person_tracking_counter += 1
        
        # Keep unmatched tracks for a few frames
        for person_id, person_data in self.tracked_persons.items():
            if person_id not in updated_tracks:
                person_data['frames_since_update'] += 1
                if person_data['frames_since_update'] < 5:
                    updated_tracks[person_id] = person_data
        
        self.tracked_persons = updated_tracks

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area

    def draw_detections(self, frame):
        """Draw all detections on the frame"""
        if frame is None or frame.size == 0:
            return frame

        height, width = frame.shape[:2]
        
        # Draw tracked faces
        for track_id, track_data in self.tracked_faces.items():
            box = track_data['box']
            left, top, right, bottom = box
            
            cv2.rectangle(frame, (left, top), (right, bottom), 
                         self.label_colors['face'], 2)
            cv2.rectangle(frame, (left, top-30), (right, top), 
                         self.label_colors['face'], -1)
            label = f"{track_data['name']} ({track_data['confidence']:.0f}%)"
            cv2.putText(frame, label,
                       (left+5, top-10), self.font, 0.6, (255, 255, 255), 1)
        
        # Draw tracked persons with motion status
        for person_id, person_data in self.tracked_persons.items():
            box = person_data['box']
            left, top, right, bottom = box
            
            motion_status = "Moving" if person_data['is_moving'] else "Idle"
            motion_color = self.label_colors['moving'] if person_data['is_moving'] else self.label_colors['idle']
            
            # Draw person bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), motion_color, 2)
            
            # Create background for labels
            cv2.rectangle(frame, (left, top-30), (right, top), motion_color, -1)
            
            # Show motion status
            cv2.putText(frame, f"Person: {motion_status}",
                       (left+5, top-10), self.font, 0.6, (255, 255, 255), 1)
            
            # Draw motion trail if moving
            if person_data['is_moving'] and len(person_data['position_history']) > 1:
                # Draw trail points
                for i in range(1, len(person_data['position_history'])):
                    pt1 = person_data['position_history'][i-1]
                    pt2 = person_data['position_history'][i]
                    cv2.line(frame, pt1, pt2, motion_color, 2)
        
        # Draw object labels (excluding "Person" as we handle them separately)
        label_positions = {}
        for label_name, boxes in self.last_results['objects'].items():
            for box in boxes:
                left = int(box['box'][0])
                top = int(box['box'][1])
                box_width = int(box['box'][2] - box['box'][0])
                box_height = int(box['box'][3] - box['box'][1])
                
                (text_width, text_height), _ = cv2.getTextSize(
                    label_name, self.font, self.font_scale, self.font_thickness)
                
                position_key = f"{left}_{top}"
                y_offset = label_positions.get(position_key, 0)
                
                cv2.rectangle(frame, 
                             (left, top), 
                             (left + box_width, top + box_height),
                             self.label_colors['object'], 2)
                
                label_y = top + y_offset - 5 if (top - y_offset) > 20 else top + box_height + y_offset + 20
                cv2.rectangle(frame,
                             (left, label_y - text_height - 5),
                             (left + text_width + 5, label_y + 5),
                             (50, 50, 50), -1)
                cv2.putText(frame, label_name,
                           (left + 3, label_y),
                           self.font, self.font_scale,
                           (255, 255, 255), self.font_thickness)
                
                label_positions[position_key] = y_offset + text_height + 15
        
        return frame

def main():
    print("=== YOLO VIDEO ANALYZER ===")
    video_path = "clip1.mp4"
    
    analyzer = VideoAnalyzer()
    analyzer.process_video(f"assets/input_videos/{video_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Script terminated by user")
    except Exception as e:
        print(f"❌ Error: {e}")
