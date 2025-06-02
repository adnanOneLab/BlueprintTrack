import cv2
import cv2
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget

class VideoHandlerMixin:
    def _init_video_handler(self):
        self.video_capture = None
        self.current_frame = None
        self.scaled_frame = None
        self.fps = 30
        self.frame_offset_x = 0
        self.frame_offset_y = 0
        self.scale_factor = 1.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

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
            self.aws_service.last_detection_time = 0
            return True
        return False

    def update_frame(self):
        """Update the current frame from video"""
        if self.video_capture is None:
            return
        
        ret, frame = self.video_capture.read()
        if not ret:
            if self.is_exporting:
                self.is_exporting = False
                self.status_message.emit("Export complete")
            return
        
        # Convert to RGB for display
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale frame to fit widget while maintaining aspect ratio
        self.update_scaled_frame()
        
        # Only process frames and update tracking during export
        if self.is_exporting:
            current_time = self.frame_number / self.fps
            
            # Process person detection if AWS is enabled
            if self.aws_service.aws_enabled:
                try:
                    detected_people = self.aws_service.detect_people(frame, current_time)
                    if detected_people:
                        # Update tracking without clearing tracked_people
                        self.tracked_people = self.person_tracker.update(detected_people, self.stores, self.frame_number)
                    else:
                        # If no detections, still update tracking to clean up old tracks
                        self.tracked_people = self.person_tracker.update([], self.stores, self.frame_number)
                except Exception as e:
                    print(f"Error in person detection: {str(e)}")
        
        self.frame_number += 1
        self.update()  # Ensure the widget is updated to show new frame

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
        QWidget.resizeEvent(self, event)
        self.update_scaled_frame()
        self.update()