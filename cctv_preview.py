import cv2
import numpy as np
import os
import boto3
from botocore.exceptions import ClientError
from PyQt6.QtWidgets import (QWidget, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QFont
from datetime import datetime

from person_tracker import PersonTracker

class CCTVPreview(QWidget):
    """Widget for displaying and testing CCTV footage"""
    calibration_points_selected = pyqtSignal(list)  # List of (x, y) points
    status_message = pyqtSignal(str)  # Signal for status updates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_capture = None
        self.current_frame = None
        self.scaled_frame = None
        self.stores = {}
        self.cameras = {}
        self.calibration_points = []
        self.store_perspective_matrices = {}  # Store individual matrices for each store
        self.calibration_mode = False
        self.test_mode = False
        self.calibration_point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.setMinimumSize(640, 480)
        self.setToolTip("CCTV footage preview with store detection")
        
        # AWS Rekognition setup
        self.rekognition_client = None
        self.aws_enabled = False
        self.detected_people = []  # List of current detected people with bounding boxes
        self.frame_count = 0
        self.last_detection_time = 0  # Track last detection time
        self.detection_interval = 3.0  # Process every 1 second
        self.fps = 30  # Default FPS, will be updated when video is loaded
        
        # Store entry notification setup
        self.last_store_entry = None  # Track the last store entry
        self.store_entry_display_time = 0  # How long to show the entry message (in frames)
        self.notification_duration = 30  # Show notification for 30 frames (about 1 second)
        
        # Set size policy to allow widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set up video update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
        
        self.video_writer = None
        self.is_exporting = False
        self.export_progress = 0
        self.total_frames = 0
        
        # Add person tracker
        self.person_tracker = PersonTracker()
        self.frame_number = 0
        
        # Track API usage for cost management
        self.api_calls_count = 0
        self.last_api_call_time = None
    
    def load_video(self, video_path):
        """Load a video file for testing"""
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            return False
        
        # Get video FPS and update detection interval
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default to 30 FPS if not available
        
        # Read and display first frame
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_scaled_frame()
            self.update()
            # Reset to first frame
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_number = 0
            self.last_detection_time = 0
            self.api_calls_count = 0
            self.last_api_call_time = None
            return True
        return False
    
    def update_frame(self):
        """Update the current frame from video"""
        if self.video_capture is None or not self.is_exporting:
            return
        
        ret, frame = self.video_capture.read()
        if not ret:
            self.is_exporting = False
            self.status_message.emit("Export complete")
            return
        
        # Convert to RGB for display
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale frame to fit widget while maintaining aspect ratio
        self.update_scaled_frame()
        
        # Calculate time since last detection
        current_time = self.frame_number / self.fps
        time_since_last_detection = current_time - self.last_detection_time
        
        # Process person detection if AWS is enabled and enough time has passed
        if self.aws_enabled and time_since_last_detection >= self.detection_interval:
            try:
                detected_people = self.detect_people(frame)
                self.last_detection_time = current_time
                self.api_calls_count += 1
                self.last_api_call_time = datetime.now()
                
                # Update person tracking
                self.person_tracker.update(detected_people, self.stores, self.frame_number)
                
                # Log API usage
                if self.api_calls_count % 10 == 0:  # Log every 10 calls
                    print(f"AWS API calls: {self.api_calls_count} (Last call: {self.last_api_call_time})")
            except Exception as e:
                print(f"Error in person detection: {str(e)}")
        
        # Apply perspective transformation if available
        if self.stores:
            try:
                # Transform store polygons to video coordinates
                for store_id, store in self.stores.items():
                    if "polygon" in store and len(store["polygon"]) > 2:
                        perspective_matrix = self.store_perspective_matrices.get(store_id)
                        if perspective_matrix is None:
                            continue
                            
                        points = np.array(store["polygon"], dtype=np.float32).reshape(-1, 1, 2)
                        if perspective_matrix.shape != (3, 3):
                            continue
                        
                        transformed = cv2.perspectiveTransform(points, perspective_matrix)
                        store["video_polygon"] = [(int(p[0][0]), int(p[0][1])) for p in transformed]
            except Exception as e:
                print(f"Error in perspective transformation: {str(e)}")
        
        self.frame_number += 1
        self.update()
    
    def detect_people(self, frame):
        """Detect people in the frame using AWS Rekognition"""
        if not self.aws_enabled or self.rekognition_client is None:
            return []
        
        try:
            # Convert frame to JPEG bytes with reduced quality for cost optimization
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Reduce quality to 85%
            _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_param)
            
            # Call AWS Rekognition with optimized parameters
            response = self.rekognition_client.detect_labels(
                Image={'Bytes': jpeg_bytes.tobytes()},
                MaxLabels=5,  # Reduced from 10 to 5 since we only care about people
                MinConfidence=75.0  # Increased confidence threshold
            )
            
            # Filter for people and extract bounding boxes
            people = []
            for label in response['Labels']:
                if label['Name'].lower() == 'person':
                    for instance in label.get('Instances', []):
                        if instance['Confidence'] >= 75.0:  # Increased confidence threshold
                            bbox = instance['BoundingBox']
                            # Convert normalized coordinates to pixel coordinates
                            x = int(bbox['Left'] * frame.shape[1])
                            y = int(bbox['Top'] * frame.shape[0])
                            width = int(bbox['Width'] * frame.shape[1])
                            height = int(bbox['Height'] * frame.shape[0])
                            confidence = instance['Confidence']
                            people.append({
                                'bbox': (x, y, width, height),
                                'confidence': confidence
                            })
            
            return people
            
        except Exception as e:
            print(f"Error in person detection: {str(e)}")
            return []
    
    def enable_aws_rekognition(self, aws_region='ap-south-1'):
        """Enable AWS Rekognition using default credentials"""
        try:
            # Use default credentials with specified region
            self.rekognition_client = boto3.client('rekognition', region_name=aws_region)
            
            # Test the connection with minimal API call
            self.rekognition_client.detect_labels(
                Image={'Bytes': cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()},
                MaxLabels=1,
                MinConfidence=90.0
            )
            
            self.aws_enabled = True
            self.api_calls_count = 0
            self.last_api_call_time = None
            self.status_message.emit("AWS Rekognition enabled successfully")
            print("AWS Rekognition enabled - API calls will be made every 1 second")
            return True
            
        except ClientError as e:
            self.aws_enabled = False
            self.rekognition_client = None
            self.status_message.emit(f"AWS Rekognition error: {str(e)}")
            return False
        except Exception as e:
            self.aws_enabled = False
            self.rekognition_client = None
            self.status_message.emit(f"Error enabling AWS Rekognition: {str(e)}")
            return False
    
    def update_scaled_frame(self):
        """Scale the current frame to fit the widget while maintaining aspect ratio"""
        if self.current_frame is None:
            return
            
        # Get widget size
        widget_size = self.size()
        
        # Calculate scaling factors
        frame_height, frame_width = self.current_frame.shape[:2]
        scale_w = widget_size.width() / frame_width
        scale_h = widget_size.height() / frame_height
        self.scale_factor = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(frame_width * self.scale_factor)
        new_height = int(frame_height * self.scale_factor)
        
        # Resize frame
        self.scaled_frame = cv2.resize(self.current_frame, (new_width, new_height))
        
        # Calculate offset to center the frame
        self.frame_offset_x = (widget_size.width() - new_width) // 2
        self.frame_offset_y = (widget_size.height() - new_height) // 2
    
    def resizeEvent(self, event):
        """Handle widget resize events"""
        super().resizeEvent(event)
        self.update_scaled_frame()
        self.update()
    
    def paintEvent(self, event):
        """Draw the video frame and overlays"""
        if self.scaled_frame is None:
            return
        
        painter = QPainter(self)
        
        # Draw video frame
        height, width = self.scaled_frame.shape[:2]
        qimage = QImage(self.scaled_frame.data, width, height,
                       self.scaled_frame.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(self.frame_offset_x, self.frame_offset_y, qimage)
        
        # Draw tracked people (only those in stores)
        if self.aws_enabled and hasattr(self, 'tracked_people'):
            for person_id, person in self.tracked_people.items():
                if person['current_store'] is not None:  # Only draw if person is in a store
                    x, y, w, h = person['bbox']
                    current_store = person['current_store']
                    store_name = self.stores[current_store]['name']
                    
                    # Scale coordinates to widget size
                    scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                    scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                    scaled_w = int(w * self.scale_factor)
                    scaled_h = int(h * self.scale_factor)
                    
                    # Draw bounding box with thinner line
                    painter.setPen(QPen(QColor(255, 0, 0), 1))
                    painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
                    
                    # Draw small ID and store info
                    label = f"{person_id}|{store_name[:5]}"  # Truncate store name to 5 chars
                    painter.setFont(QFont("Arial", 12))  # Smaller font size
                    painter.setPen(QColor(255, 255, 255))
                    # Draw text background
                    text_rect = painter.fontMetrics().boundingRect(label)
                    text_rect.moveTop(scaled_y - text_rect.height())
                    text_rect.moveLeft(scaled_x)
                    text_rect.adjust(-1, -1, 1, 1)  # Smaller padding
                    painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                    # Draw text
                    painter.drawText(scaled_x, scaled_y - 2, label)  # Reduced vertical offset
        
        # Draw calibration points and lines if in calibration mode
        if self.calibration_mode:
            # Draw existing points
            for i, point in enumerate(self.calibration_points):
                x = int(point[0] * self.scale_factor + self.frame_offset_x)
                y = int(point[1] * self.scale_factor + self.frame_offset_y)
                
                # Draw point
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawEllipse(QPoint(x, y), 5, 5)
                
                # Draw label
                painter.setFont(QFont("Arial", 17, QFont.Weight.Bold))
                painter.setPen(QColor(255, 255, 255))
                # Draw text background
                text = f"{i+1}. {self.calibration_point_labels[i]}"
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(x, y - 20))
                text_rect.adjust(-5, -2, 5, 2)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(x, y - 20, text)
            
            # Draw lines between points
            if len(self.calibration_points) > 1:
                painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
                for i in range(len(self.calibration_points) - 1):
                    x1 = int(self.calibration_points[i][0] * self.scale_factor + self.frame_offset_x)
                    y1 = int(self.calibration_points[i][1] * self.scale_factor + self.frame_offset_y)
                    x2 = int(self.calibration_points[i+1][0] * self.scale_factor + self.frame_offset_x)
                    y2 = int(self.calibration_points[i+1][1] * self.scale_factor + self.frame_offset_y)
                    painter.drawLine(x1, y1, x2, y2)
                
                # Draw line from last point to first point if we have 3 points
                if len(self.calibration_points) == 3:
                    x1 = int(self.calibration_points[-1][0] * self.scale_factor + self.frame_offset_x)
                    y1 = int(self.calibration_points[-1][1] * self.scale_factor + self.frame_offset_y)
                    x2 = int(self.calibration_points[0][0] * self.scale_factor + self.frame_offset_x)
                    y2 = int(self.calibration_points[0][1] * self.scale_factor + self.frame_offset_y)
                    painter.drawLine(x1, y1, x2, y2)
            
            # Draw next point indicator
            if len(self.calibration_points) < 4:
                next_point = self.calibration_point_labels[len(self.calibration_points)]
                painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                painter.setPen(QColor(255, 255, 255))
                # Draw text background
                text = f"Click to place {next_point} point"
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(self.width() // 2, 30))
                text_rect.adjust(-10, -5, 10, 5)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(self.width() // 2, 30, text)
        
        # Draw store polygons in test mode
        if self.test_mode and self.stores:
            # Debug information
            print(f"Drawing {len(self.stores)} stores in test mode")
            
            for store_id, store in self.stores.items():
                if "video_polygon" in store and len(store["video_polygon"]) > 2:
                    try:
                        # Scale the transformed points to widget coordinates
                        video_polygon = []
                        for x, y in store["video_polygon"]:
                            scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                            scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                            video_polygon.append((scaled_x, scaled_y))
                        
                        if len(video_polygon) > 2:
                            # Draw filled semi-transparent polygon
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.setBrush(QColor(0, 255, 0, 50))  # Semi-transparent green
                            painter.drawPolygon([QPoint(x, y) for x, y in video_polygon])
                            
                            # Draw polygon outline
                            painter.setPen(QPen(QColor(0, 255, 0), 2))
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            for i in range(len(video_polygon) - 1):
                                painter.drawLine(video_polygon[i][0], video_polygon[i][1],
                                               video_polygon[i+1][0], video_polygon[i+1][1])
                            painter.drawLine(video_polygon[-1][0], video_polygon[-1][1],
                                           video_polygon[0][0], video_polygon[0][1])
                            
                            # Draw store name
                            if "name" in store:
                                centroid_x = int(np.mean([p[0] for p in video_polygon]))
                                centroid_y = int(np.mean([p[1] for p in video_polygon]))
                                painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
                                # Draw text background
                                text_rect = painter.fontMetrics().boundingRect(store["name"])
                                text_rect.moveCenter(QPoint(centroid_x, centroid_y))
                                text_rect.adjust(-5, -2, 5, 2)
                                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                                # Draw text
                                painter.setPen(QColor(255, 255, 255))
                                painter.drawText(centroid_x, centroid_y, store["name"])
                    except Exception as e:
                        print(f"Error drawing store {store_id}: {str(e)}")
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for calibration"""
        if not self.calibration_mode:
            return
        
        # Convert widget coordinates to original video coordinates
        x = (event.position().x() - self.frame_offset_x) / self.scale_factor
        y = (event.position().y() - self.frame_offset_y) / self.scale_factor
        
        # Check if click is within video bounds
        if not (0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]):
            QMessageBox.warning(self, "Invalid Point", 
                "Please click within the video frame.")
            return
        
        if len(self.calibration_points) < 4:
            # Add point with validation
            if len(self.calibration_points) > 0:
                # Check if point is too close to existing points
                for existing_point in self.calibration_points:
                    distance = np.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                    if distance < 20:  # Minimum 20 pixels between points
                        QMessageBox.warning(self, "Invalid Point", 
                            "Please click further away from existing points.")
                        return
            
            self.calibration_points.append((x, y))
            if len(self.calibration_points) == 4:
                # Validate the quadrilateral
                points = np.array(self.calibration_points)
                # Check if points form a reasonable quadrilateral
                if not self.validate_quadrilateral(points):
                    QMessageBox.warning(self, "Invalid Points", 
                        "The selected points do not form a valid quadrilateral.\n"
                        "Please try again, making sure to click points in the correct order:\n"
                        "1. Top-Left\n2. Top-Right\n3. Bottom-Right\n4. Bottom-Left")
                    self.calibration_points = []
                    return
                self.calibration_points_selected.emit(self.calibration_points)
            self.update()

    def validate_quadrilateral(self, points):
        """Validate that the points form a reasonable quadrilateral"""
        try:
            # Calculate distances between consecutive points
            distances = []
            for i in range(4):
                j = (i + 1) % 4
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
            
            # Check if any side is too short
            if min(distances) < 20:  # Minimum 20 pixels
                return False
            
            # Check if the quadrilateral is too skewed
            # Calculate angles between consecutive sides
            angles = []
            for i in range(4):
                j = (i + 1) % 4
                k = (i + 2) % 4
                v1 = points[j] - points[i]
                v2 = points[k] - points[j]
                angle = np.abs(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                angles.append(np.degrees(angle))
            
            # Check if any angle is too small or too large
            if min(angles) < 30 or max(angles) > 150:  # Angles should be between 30 and 150 degrees
                return False
            
            return True
        except Exception as e:
            print(f"Error validating quadrilateral: {str(e)}")
            return False

    def calculate_perspective_transform(self, blueprint_points, video_points, store_id):
        """Calculate perspective transformation matrix for a specific store"""
        if len(blueprint_points) != 4 or len(video_points) != 4:
            return False
        
        try:
            # Convert points to numpy arrays
            src_points = np.array(blueprint_points, dtype=np.float32)
            dst_points = np.array(video_points, dtype=np.float32)
            
            # Calculate perspective transform matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Basic validation that the matrix is valid
            if perspective_matrix is None or perspective_matrix.shape != (3, 3):
                print(f"Error: Failed to create valid perspective matrix for store {store_id}")
                return False
            
            # Store the matrix for this specific store
            self.store_perspective_matrices[store_id] = perspective_matrix
            print(f"Perspective transformation matrix created successfully for store {store_id}")
            return True
            
        except Exception as e:
            print(f"Error in perspective transformation for store {store_id}: {str(e)}")
            return False

    def set_calibration_mode(self, enabled):
        """Enable/disable calibration point selection"""
        self.calibration_mode = enabled
        self.calibration_points = []  # Clear existing calibration points
        if enabled:
            self.setToolTip("Click 4 points in the video in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        else:
            self.setToolTip("CCTV footage preview with store detection")
        self.update()

    def set_test_mode(self, enabled):
        """Enable/disable test mode for store detection"""
        self.test_mode = enabled
        if enabled:
            # Debug information
            print(f"Test mode enabled. Stores: {len(self.stores)}, Perspective matrices: {len(self.store_perspective_matrices)}")
            if not self.stores:
                print("Warning: No stores defined")
            self.setToolTip("Test mode: Store boundaries are highlighted in video")
        else:
            self.setToolTip("CCTV footage preview with store detection")
        self.update()

    def export_video(self, output_path):
        """Export the processed video with drawings"""
        if not self.video_capture or self.current_frame is None:
            self.status_message.emit("Error: No video loaded")
            return False
        
        try:
            # Get video properties
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps  # Duration in seconds
            
            print(f"Input video properties:")
            print(f"- Resolution: {frame_width}x{frame_height}")
            print(f"- FPS: {fps}")
            print(f"- Total frames: {total_frames}")
            print(f"- Duration: {duration:.2f} seconds")
            
            # Store current position to restore later
            current_position = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Create video writer with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
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
                    # Create a copy for drawing
                    frame_draw = frame.copy()
                    
                    # Apply perspective transformation for stores
                    if self.stores:
                        try:
                            # Transform store polygons to video coordinates
                            for store_id, store in self.stores.items():
                                if "polygon" in store and len(store["polygon"]) > 2:
                                    perspective_matrix = self.store_perspective_matrices.get(store_id)
                                    if perspective_matrix is None:
                                        continue
                                    
                                    points = np.array(store["polygon"], dtype=np.float32).reshape(-1, 1, 2)
                                    if perspective_matrix.shape != (3, 3):
                                        continue
                                    
                                    transformed = cv2.perspectiveTransform(points, perspective_matrix)
                                    store["video_polygon"] = [(int(p[0][0]), int(p[0][1])) for p in transformed]
                        except Exception as e:
                            print(f"Error in perspective transformation: {str(e)}")
                    
                    # Draw store polygons
                    if self.stores:
                        for store_id, store in self.stores.items():
                            if "video_polygon" in store and len(store["video_polygon"]) > 2:
                                try:
                                    points = np.array(store["video_polygon"], np.int32).reshape((-1, 1, 2))
                                    
                                    # Draw filled semi-transparent polygon
                                    overlay = frame_draw.copy()
                                    cv2.fillPoly(overlay, [points], (0, 255, 0))
                                    cv2.addWeighted(overlay, 0.3, frame_draw, 0.7, 0, frame_draw)
                                    
                                    # Draw polygon outline
                                    cv2.polylines(frame_draw, [points], True, (0, 255, 0), 1)
                                    
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
                    
                    # Process person detection if AWS is enabled
                    if self.aws_enabled:
                        try:
                            detected_people = self.detect_people(frame)
                            tracked_people = self.person_tracker.update(detected_people, self.stores, self.frame_number)
                            
                            # Draw tracked people (only those in stores)
                            for person_id, person in tracked_people.items():
                                if person['current_store'] is not None:
                                    x, y, w, h = person['bbox']
                                    current_store = person['current_store']
                                    store_name = self.stores[current_store]['name']
                                    
                                    cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
                                    
                                    label = f"{person_id}|{store_name[:5]}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.8
                                    thickness = 1
                                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                                    
                                    cv2.rectangle(frame_draw, 
                                                (x, y - text_height - 2),
                                                (x + text_width, y),
                                                (0, 0, 0), -1)
                                    
                                    cv2.putText(frame_draw, label,
                                              (x, y - 2),
                                              font, font_scale, (255, 255, 255), thickness)
                            
                            # Draw store entry notification
                            if self.last_store_entry and self.store_entry_display_time > 0:
                                text = f"Person {self.last_store_entry['person_id']} entered {self.last_store_entry['store_name']}"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.8
                                thickness = 1
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                
                                cv2.rectangle(frame_draw,
                                            (frame_draw.shape[1]//2 - text_width//2 - 5, 10),
                                            (frame_draw.shape[1]//2 + text_width//2 + 5, 10 + text_height + 5),
                                            (0, 0, 0), -1)
                                
                                cv2.putText(frame_draw, text,
                                          (frame_draw.shape[1]//2 - text_width//2, 10 + text_height),
                                          font, font_scale, (255, 255, 255), thickness)
                                
                                self.store_entry_display_time -= 1
                        except Exception as e:
                            print(f"Error in person detection/tracking: {str(e)}")
                    
                    # Write frame
                    self.video_writer.write(frame_draw)
                    processed_frames += 1
                    
                    # Update progress
                    frame_count += 1
                    progress = (frame_count / total_frames) * 100
                    self.status_message.emit(f"Exporting video: {progress:.1f}%")
                    
                    self.frame_number += 1
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    continue
            
            # Clean up
            self.video_writer.release()
            self.video_writer = None
            self.is_exporting = False
            self.export_progress = 0
            
            # Verify output video
            output_cap = cv2.VideoCapture(output_path)
            if output_cap.isOpened():
                output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_fps = output_cap.get(cv2.CAP_PROP_FPS)
                output_duration = output_frames / output_fps
                output_cap.release()
                
                print(f"Output video properties:")
                print(f"- Resolution: {frame_width}x{frame_height}")
                print(f"- FPS: {output_fps}")
                print(f"- Total frames: {output_frames}")
                print(f"- Duration: {output_duration:.2f} seconds")
                print(f"- Processed frames: {processed_frames}")
                
                # Verify frame count
                if output_frames != total_frames:
                    print(f"Error: Frame count mismatch - Input: {total_frames}, Output: {output_frames}, Processed: {processed_frames}")
                    # Delete the incorrect output file
                    os.remove(output_path)
                    raise Exception(f"Frame count mismatch in output video (Input: {total_frames}, Output: {output_frames}, Processed: {processed_frames})")
                
                if abs(output_duration - duration) > 0.1:
                    print(f"Error: Duration mismatch - Input: {duration:.2f}s, Output: {output_duration:.2f}s")
                    # Delete the incorrect output file
                    os.remove(output_path)
                    raise Exception(f"Duration mismatch in output video (Input: {duration:.2f}s, Output: {output_duration:.2f}s)")
            else:
                print("Warning: Could not verify output video properties")
                raise Exception("Could not verify output video")
            
            # Restore video capture position
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            self.update_frame()
            
            self.status_message.emit(f"Video exported successfully to: {output_path}")
            return True
            
        except Exception as e:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_exporting = False
            self.export_progress = 0
            # Restore video capture position on error
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            self.status_message.emit(f"Error exporting video: {str(e)}")
            print(f"Export error details: {str(e)}")
            return False