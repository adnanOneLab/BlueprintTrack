import cv2
import boto3
import time
import os
import numpy as np
from collections import defaultdict
import base64
from io import BytesIO
from PIL import Image

class RekognitionVideoAnalyzer:
    def __init__(self, region_name='ap-south-1'):
        # Initialize AWS Rekognition client
        self.rekognition = boto3.client('rekognition', region_name=region_name)
        
        # Visualization settings
        self.label_colors = {
            'person': (0, 255, 0),    # Green for all persons
            'idle': (255, 165, 0),    # Orange for idle persons
            'moving': (255, 0, 255)   # Purple for moving persons
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
        # Processing control
        self.frame_skip = 5  # Process every 5th frame for Rekognition (API cost optimization)
        self.frame_counter = 0
        self.last_results = {'persons': []}
        
        # Person tracking (keeping your excellent tracking logic)
        self.tracked_persons = {}
        self.person_tracking_counter = 0
        self.motion_threshold = 8
        self.iou_threshold = 0.25
        self.max_frames_without_detection = 15  # Longer since Rekognition calls are less frequent
        self.min_track_length = 5
        
        # Rekognition settings
        self.rekognition_confidence_threshold = 80  # Higher confidence for better quality
        self.max_image_size = (800, 600)  # Resize for faster API calls and cost optimization

    def frame_to_bytes(self, frame):
        """Convert OpenCV frame to bytes for Rekognition API"""
        # Resize frame if too large (for cost and speed optimization)
        height, width = frame.shape[:2]
        if width > self.max_image_size[0] or height > self.max_image_size[1]:
            scale = min(self.max_image_size[0]/width, self.max_image_size[1]/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to bytes
        pil_image = Image.fromarray(frame_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue(), frame.shape[:2]  # Return bytes and original frame dimensions

    def detect_persons_with_rekognition(self, frame):
        """Use AWS Rekognition to detect persons in the frame"""
        try:
            # Convert frame to bytes
            image_bytes, (orig_height, orig_width) = self.frame_to_bytes(frame)
            
            # Call Rekognition API
            response = self.rekognition.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=50,
                MinConfidence=self.rekognition_confidence_threshold,
                Features=['GENERAL_LABELS']
            )
            
            persons = []
            
            # Process labels to find persons
            for label in response['Labels']:
                if label['Name'].lower() == 'person':
                    # Get bounding boxes for person instances
                    for instance in label.get('Instances', []):
                        if 'BoundingBox' in instance:
                            bbox = instance['BoundingBox']
                            confidence = instance['Confidence']
                            
                            # Convert relative coordinates to absolute
                            frame_height, frame_width = frame.shape[:2]
                            left = int(bbox['Left'] * frame_width)
                            top = int(bbox['Top'] * frame_height)
                            width = int(bbox['Width'] * frame_width)
                            height = int(bbox['Height'] * frame_height)
                            right = left + width
                            bottom = top + height
                            
                            # Ensure coordinates are within frame bounds
                            left = max(0, left)
                            top = max(0, top)
                            right = min(frame_width, right)
                            bottom = min(frame_height, bottom)
                            
                            # Skip invalid boxes
                            if right <= left or bottom <= top:
                                continue
                            
                            persons.append({
                                'box': (left, top, right, bottom),
                                'confidence': confidence / 100.0,  # Convert to 0-1 range
                                'center': ((left + right) // 2, (top + bottom) // 2)
                            })
            
            return {'persons': persons}
            
        except Exception as e:
            print(f"⚠️ Rekognition API error: {e}")
            # Fallback to empty detection
            return {'persons': []}

    def process_video(self, video_path):
        """Process a video file with Rekognition + custom tracking"""
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
        
        print(f"\n📹 Processing video with AWS Rekognition: {video_path}")
        print(f"📏 Resolution: {width}x{height}")
        print(f"🎞️ FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"🔄 Processing every {self.frame_skip} frames with Rekognition")
        print("🛑 Press 'q' to stop processing early\n")

        # Create output file
        output_path = video_path.split("/")[-1].split(".")[0] + '_rekognition_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_time = time.time()
        processed_frames = 0
        api_calls = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                self.frame_counter += 1
                continue

            self.frame_counter += 1
            
            # Use Rekognition every N frames, track on every frame
            if self.frame_counter % self.frame_skip == 0:
                processed_frames += 1
                api_calls += 1
                print(f"🔍 Rekognition API call #{api_calls} (Frame {self.frame_counter})")
                
                self.last_results = self.detect_persons_with_rekognition(frame)
                self.update_tracked_persons(frame)
            else:
                # Still update tracking for smoother motion detection
                self.update_person_positions()
            
            # Draw detections on frame
            frame_with_detections = self.draw_detections(frame.copy())
            
            # Display processing info
            person_count = len(self.tracked_persons)
            moving_count = sum(1 for p in self.tracked_persons.values() if p['is_moving'])
            idle_count = person_count - moving_count
            
            info_text = f"Frame: {self.frame_counter}/{total_frames} | API Calls: {api_calls} | Persons: {person_count} (Moving: {moving_count}, Idle: {idle_count})"
            cv2.putText(frame_with_detections, info_text, (10, 30), self.font, 0.6, (255, 255, 255), 2)
            
            # Write frame to output
            out.write(frame_with_detections)
            
            # Display preview
            display_frame = frame_with_detections
            if width > 1280:
                display_width = 1280
                display_height = int(height * (display_width / width))
                display_frame = cv2.resize(frame_with_detections, (display_width, display_height))
            
            cv2.imshow("Rekognition + Custom Tracking (Press Q to stop)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        processing_time = time.time() - start_time
        estimated_cost = api_calls * 0.001  # Rough estimate: $1 per 1000 API calls
        
        print(f"\n✅ Analysis complete!")
        print(f"⏱️ Processed {processed_frames} frames in {processing_time:.1f} seconds")
        print(f"🔌 Made {api_calls} Rekognition API calls")
        print(f"💰 Estimated cost: ~${estimated_cost:.3f}")
        print(f"💾 Output saved to: {output_path}")
        print(f"👥 Total unique persons tracked: {self.person_tracking_counter}")

    # Keep all your excellent tracking methods unchanged
    def update_tracked_persons(self, frame):
        """Track all persons and detect if they're moving or idle"""
        current_detections = self.last_results['persons']
        
        updated_tracks = {}
        matched_detections = set()
        
        # Hungarian algorithm approach - find best overall matching
        detection_track_pairs = []
        
        for i, detection in enumerate(current_detections):
            detection_box = detection['box']
            detection_center = detection['center']
            
            for person_id, person_data in self.tracked_persons.items():
                if person_data['frames_since_update'] > self.max_frames_without_detection:
                    continue
                
                prev_box = person_data['box']
                prev_center = person_data['center']
                
                iou = self.calculate_iou(detection_box, prev_box)
                center_distance = np.sqrt((detection_center[0] - prev_center[0])**2 + 
                                        (detection_center[1] - prev_center[1])**2)
                
                if iou > self.iou_threshold:
                    score = iou - (center_distance / 1000)
                    detection_track_pairs.append((i, person_id, score, detection))
        
        # Sort by score and greedily assign
        detection_track_pairs.sort(key=lambda x: x[2], reverse=True)
        used_detections = set()
        used_tracks = set()
        
        for det_idx, person_id, score, detection in detection_track_pairs:
            if det_idx in used_detections or person_id in used_tracks:
                continue
            
            person_data = self.tracked_persons[person_id]
            detection_box = detection['box']
            detection_center = detection['center']
            
            position_history = person_data.get('position_history', [])
            position_history.append(detection_center)
            if len(position_history) > 20:
                position_history = position_history[-20:]
            
            is_moving, avg_movement, motion_confidence = self.calculate_movement_improved(
                position_history, person_data.get('track_length', 0))
            
            updated_tracks[person_id] = {
                'box': detection_box,
                'center': detection_center,
                'confidence': detection['confidence'],
                'frames_since_update': 0,
                'position_history': position_history,
                'is_moving': is_moving,
                'avg_movement': avg_movement,
                'motion_confidence': motion_confidence,
                'track_length': person_data.get('track_length', 0) + 1,
                'last_seen_frame': self.frame_counter
            }
            
            used_detections.add(det_idx)
            used_tracks.add(person_id)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(current_detections):
            if i not in used_detections:
                new_id = f"person_{self.person_tracking_counter}"
                updated_tracks[new_id] = {
                    'box': detection['box'],
                    'center': detection['center'],
                    'confidence': detection['confidence'],
                    'frames_since_update': 0,
                    'position_history': [detection['center']],
                    'is_moving': False,
                    'avg_movement': 0,
                    'motion_confidence': 0,
                    'track_length': 1,
                    'last_seen_frame': self.frame_counter
                }
                self.person_tracking_counter += 1
        
        # Keep unmatched tracks for longer (since Rekognition calls are less frequent)
        for person_id, person_data in self.tracked_persons.items():
            if person_id not in updated_tracks:
                person_data['frames_since_update'] += 1
                if person_data['frames_since_update'] <= self.max_frames_without_detection:
                    predicted_center = self.predict_position(person_data)
                    if predicted_center:
                        person_data['center'] = predicted_center
                        old_box = person_data['box']
                        box_width = old_box[2] - old_box[0]
                        box_height = old_box[3] - old_box[1]
                        person_data['box'] = (
                            predicted_center[0] - box_width//2,
                            predicted_center[1] - box_height//2,
                            predicted_center[0] + box_width//2,
                            predicted_center[1] + box_height//2
                        )
                    
                    person_data['confidence'] *= 0.95  # Slower confidence decay
                    updated_tracks[person_id] = person_data
        
        self.tracked_persons = updated_tracks

    def update_person_positions(self):
        """Update position tracking for frames where we don't run detection"""
        for person_id, person_data in self.tracked_persons.items():
            person_data['frames_since_update'] += 1

    def calculate_movement_improved(self, position_history, track_length):
        """Your excellent movement calculation - keeping unchanged"""
        if len(position_history) < 3:
            return False, 0, 0
        
        if track_length < self.min_track_length:
            return False, 0, 0
        
        movements = []
        for i in range(1, len(position_history)):
            p1 = position_history[i-1]
            p2 = position_history[i]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            movements.append(distance)
        
        if not movements:
            return False, 0, 0
        
        recent_movements = movements[-3:] if len(movements) >= 3 else movements
        medium_movements = movements[-7:] if len(movements) >= 7 else movements
        
        recent_avg = sum(recent_movements) / len(recent_movements)
        medium_avg = sum(medium_movements) / len(medium_movements)
        
        movements_array = np.array(medium_movements)
        movement_std = np.std(movements_array)
        
        is_moving_recent = recent_avg > self.motion_threshold
        is_moving_medium = medium_avg > self.motion_threshold * 0.7
        is_moving_consistent = movement_std < recent_avg * 0.8
        
        motion_score = 0
        if is_moving_recent:
            motion_score += 0.4
        if is_moving_medium:
            motion_score += 0.4
        if is_moving_consistent and recent_avg > self.motion_threshold * 0.5:
            motion_score += 0.2
        
        motion_confidence = motion_score
        is_moving = motion_confidence > 0.5
        
        representative_movement = recent_avg if len(recent_movements) >= 3 else medium_avg
        
        return is_moving, representative_movement, motion_confidence

    def predict_position(self, person_data):
        """Your position prediction logic - keeping unchanged"""
        position_history = person_data.get('position_history', [])
        if len(position_history) < 2:
            return person_data.get('center')
        
        if len(position_history) >= 3:
            p1 = position_history[-3]
            p2 = position_history[-2]
            p3 = position_history[-1]
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            smooth_vx = (v1[0] + v2[0]) / 2
            smooth_vy = (v1[1] + v2[1]) / 2
            
            predicted_x = p3[0] + smooth_vx
            predicted_y = p3[1] + smooth_vy
        else:
            last_pos = position_history[-1]
            second_last_pos = position_history[-2]
            
            dx = last_pos[0] - second_last_pos[0]
            dy = last_pos[1] - second_last_pos[1]
            
            predicted_x = last_pos[0] + dx
            predicted_y = last_pos[1] + dy
        
        return (int(predicted_x), int(predicted_y))

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union - keeping unchanged"""
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
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

    def draw_detections(self, frame):
        """Draw all detections - keeping your visualization logic"""
        if frame is None or frame.size == 0:
            return frame

        for person_id, person_data in self.tracked_persons.items():
            box = person_data['box']
            left, top, right, bottom = box
            
            track_length = person_data.get('track_length', 0)
            motion_confidence = person_data.get('motion_confidence', 0)
            
            if track_length < self.min_track_length:
                motion_status = "Analyzing..."
                motion_color = (128, 128, 128)
            else:
                motion_status = "Moving" if person_data['is_moving'] else "Idle"
                motion_color = self.label_colors['moving'] if person_data['is_moving'] else self.label_colors['idle']
            
            cv2.rectangle(frame, (left, top), (right, bottom), motion_color, 3)
            
            confidence_text = ""
            if person_data.get('confidence'):
                confidence_text = f" ({person_data['confidence']:.0%})"
            
            label_text = f"{person_id}: {motion_status}{confidence_text}"
            if track_length >= self.min_track_length and motion_confidence > 0:
                label_text += f" [Rekognition]"
            
            (text_width, text_height), _ = cv2.getTextSize(label_text, self.font, 0.6, 1)
            cv2.rectangle(frame, (left, top-text_height-10), (left + text_width + 10, top), motion_color, -1)
            
            cv2.putText(frame, label_text,
                       (left+5, top-5), self.font, 0.6, (255, 255, 255), 1)
            
            # Draw motion trail
            if (person_data['is_moving'] and track_length >= self.min_track_length and 
                len(person_data['position_history']) > 1):
                points = person_data['position_history']
                recent_points = points[-10:] if len(points) > 10 else points
                
                for i in range(1, len(recent_points)):
                    alpha = i / len(recent_points)
                    pt1 = recent_points[i-1]
                    pt2 = recent_points[i]
                    
                    line_thickness = max(1, int(4 * alpha))
                    cv2.line(frame, pt1, pt2, motion_color, line_thickness)
                
                center = person_data['center']
                cv2.circle(frame, center, 5, motion_color, -1)
                cv2.circle(frame, center, 5, (255, 255, 255), 1)
        
        return frame


def main():
    print("=== AWS REKOGNITION + CUSTOM TRACKING ANALYZER ===")
    print("🌩️ Using AWS Rekognition for detection + your tracking logic")
    print("📊 Features: High-accuracy detection, motion tracking, cost optimization")
    print("\n⚙️ Setup Requirements:")
    print("1. Install boto3: pip install boto3")
    print("2. Configure AWS credentials (AWS CLI or environment variables)")
    print("3. Ensure you have Rekognition permissions\n")
    
    video_path = "clip1.mp4"
    
    # Initialize with AWS credentials (optional - will use default profile if not provided)
    analyzer = RekognitionVideoAnalyzer(
        # aws_access_key_id='your_access_key',
        # aws_secret_access_key='your_secret_key',
        # region_name='us-east-1'
    )
    
    analyzer.process_video(f"assets/input_videos/{video_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Script terminated by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()