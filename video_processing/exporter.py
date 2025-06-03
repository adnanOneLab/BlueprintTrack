import cv2
import numpy as np

class ExportMixin:
    def _init_export(self):
        self.is_exporting = False
        self.video_writer = None
        self.export_progress = 0

    def export_video(self, output_path):
        """Export the processed video with drawings"""
        if not self.video_capture or self.current_frame is None:
            self.status_message.emit("Error: No video loaded")
            return False
        
        try:
            # Verify AWS and stores setup
            if not self.aws_service.aws_enabled:
                print("Warning: AWS Rekognition is not enabled")
                return False
            
            if not self.stores:
                print("Warning: No stores defined")
                return False
            
            # Enable AWS for export mode
            self.aws_service.set_export_mode(True)
            
            # Verify store polygons and calculate video polygons
            for store_id, store in self.stores.items():
                if "polygon" not in store or len(store["polygon"]) < 3:
                    print(f"Warning: Store {store_id} has invalid polygon")
                    return False
                
                # Calculate video polygon using perspective transformation
                perspective_matrix = self.store_perspective_matrices.get(store_id)
                if perspective_matrix is None:
                    print(f"Warning: Store {store_id} has no perspective matrix")
                    return False
                
                try:
                    points = np.array(store["polygon"], dtype=np.float32).reshape(-1, 1, 2)
                    if perspective_matrix.shape != (3, 3):
                        print(f"Warning: Invalid perspective matrix shape for store {store_id}")
                        return False
                    
                    transformed = cv2.perspectiveTransform(points, perspective_matrix)
                    store["video_polygon"] = [(int(p[0][0]), int(p[0][1])) for p in transformed]
                    print(f"Calculated video polygon for store {store_id}: {store['video_polygon']}")
                except Exception as e:
                    print(f"Error calculating video polygon for store {store_id}: {str(e)}")
                    return False
            
            # Get video properties
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps  # Duration in seconds
            
            print(f"Starting video export:")
            print(f"- AWS Enabled: {self.aws_service.aws_enabled}")
            print(f"- Number of stores: {len(self.stores)}")
            print(f"- Resolution: {frame_width}x{frame_height}")
            print(f"- FPS: {fps}")
            print(f"- Total frames: {total_frames}")
            print(f"- Duration: {duration:.2f} seconds")
            
            # Store current position to restore later
            current_position = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Create video writer with MJPG codec (more widely available)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec instead of H.264
            output_path = output_path.replace('.mp4', '.avi')  # Change extension to .avi for MJPG
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not self.video_writer.isOpened():
                self.status_message.emit("Error: Could not create output video file")
                return False
            
            # Reset video capture to beginning and ensure we're at frame 0
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, _ = self.video_capture.read()  # Read first frame to ensure we're at start
            if not ret:
                raise Exception("Could not read first frame")
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset again to be sure
            
            self.is_exporting = True
            self.export_progress = 0
            self.frame_number = 0  # Reset frame counter
            
            # Process each frame
            frame_count = 0
            processed_frames = 0
            detection_count = 0
            face_detection_count = 0  # Track total face detections
            body_images_saved = 0  # Track total body images saved
            
            while frame_count < total_frames:
                # Get current frame number
                current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame != frame_count:
                    # If we're not at the expected frame, seek to it
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                
                ret, frame = self.video_capture.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_count}")
                    break
                
                try:
                    frame_draw = frame.copy()
                    
                    # Draw store polygons
                    for store_id, store in self.stores.items():
                        if "video_polygon" in store and len(store["video_polygon"]) > 2:
                            try:
                                points = np.array(store["video_polygon"], np.int32).reshape((-1, 1, 2))
                                
                                # Draw filled semi-transparent polygon
                                overlay = frame_draw.copy()
                                cv2.fillPoly(overlay, [points], (0, 255, 0))
                                cv2.addWeighted(overlay, 0.3, frame_draw, 0.7, 0, frame_draw)
                                
                                # Draw polygon outline
                                cv2.polylines(frame_draw, [points], True, (0, 255, 0), 2)
                                
                                # Draw store name
                                if "name" in store:
                                    centroid_x = int(np.mean([p[0] for p in store["video_polygon"]]))
                                    centroid_y = int(np.mean([p[1] for p in store["video_polygon"]]))
                                    
                                    text = store["name"]
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 1.3
                                    thickness = 2
                                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                    
                                    cv2.rectangle(frame_draw, 
                                                (centroid_x - text_width//2 - 2, centroid_y - text_height//2 - 2),
                                                (centroid_x + text_width//2 + 2, centroid_y + text_height//2 + 2),
                                                (0, 0, 0), -1)
                                    
                                    cv2.putText(frame_draw, text,
                                              (centroid_x - text_width//2, centroid_y + text_height//2),
                                              font, font_scale, (255, 255, 255), thickness)
                            except Exception as e:
                                print(f"Error drawing store {store_id}: {str(e)}")
                    
                    # Process person and face detection
                    current_time = frame_count / fps
                    if self.aws_service.aws_enabled:
                        try:
                            print(f"Processing frame {frame_count} at time {current_time:.2f}s")
                            # Detect people
                            detected_people = self.aws_service.detect_people(frame, current_time)
                            if detected_people:
                                detection_count += len(detected_people)
                            
                            # Detect faces
                            face_detections = self.aws_service.detect_faces(frame, current_time)
                            if face_detections:
                                face_detection_count += len(face_detections)
                                # Count body images saved
                                body_images_saved += sum(1 for face in face_detections if face.get('image_path'))
                            
                            # Update tracking with both detections
                            self.tracked_people = self.person_tracker.update(
                                detected_people, self.stores, frame_count, face_detections
                            )
                        except Exception as e:
                            print(f"Error in detection at frame {frame_count}: {str(e)}")
                    
                    # Draw tracked people
                    if self.tracked_people:
                        for person_id, person in self.tracked_people.items():
                            x, y, w, h = person['bbox']
                            current_store = person['current_store']
                            is_moving = person.get('is_moving', False)
                            
                            # Set color based on movement status
                            if is_moving:
                                box_color = (0, 165, 255)  # Orange for moving (BGR format)
                                status = "MOVING"
                            else:
                                box_color = (255, 0, 0)    # Blue for idle (BGR format)
                                status = "IDLE"
                            
                            # Draw bounding box with movement status color
                            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), box_color, 2)
                            
                            # Draw label with store info and movement status
                            if current_store is not None:
                                store_name = self.stores[current_store]['name']
                                label = f"{person_id}|{store_name[:5]}|{status}"
                            else:
                                label = f"{person_id}|OUT|{status}"
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.8
                            thickness = 2
                            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Draw text background
                            cv2.rectangle(frame_draw, 
                                        (x, y - text_height - 2),
                                        (x + text_width, y),
                                        (0, 0, 0), -1)
                            
                            # Draw text
                            cv2.putText(frame_draw, label,
                                      (x, y - 2),
                                      font, font_scale, (255, 255, 255), thickness)
                            
                            # Draw face detection if available
                            if 'face_detections' in person and person['face_detections']:
                                latest_face = person['face_detections'][-1]
                                fx, fy, fw, fh = latest_face['bbox']
                                cv2.rectangle(frame_draw, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Blue for face
                    
                    # Write frame
                    self.video_writer.write(frame_draw)
                    processed_frames += 1
                    
                    # Update progress
                    frame_count += 1
                    progress = (frame_count / total_frames) * 100
                    self.status_message.emit(f"Exporting video: {progress:.1f}%")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    continue
            
            print(f"Export completed:")
            print(f"- Total frames processed: {processed_frames}")
            print(f"- Total person detections: {detection_count}")
            print(f"- Total face detections: {face_detection_count}")
            print(f"- Body images saved: {body_images_saved}")
            print(f"- AWS API calls: {self.aws_service.api_calls_count}")
            
            # Clean up
            self.video_writer.release()
            self.video_writer = None
            self.is_exporting = False
            self.export_progress = 0
            
            # Disable AWS for preview mode
            self.aws_service.set_export_mode(False)
            
            # Restore video capture position
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            self.update_frame()
            
            self.status_message.emit(
                f"Video exported successfully to: {output_path}\n"
                f"Total person detections: {detection_count}\n"
                f"Total face detections: {face_detection_count}\n"
                f"Body images saved: {body_images_saved}"
            )
            return True
            
        except Exception as e:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_exporting = False
            self.export_progress = 0
            # Ensure AWS is disabled
            self.aws_service.set_export_mode(False)
            # Restore video capture position on error
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            self.status_message.emit(f"Error exporting video: {str(e)}")
            print(f"Export error details: {str(e)}")
            return False