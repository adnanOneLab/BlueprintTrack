# cctv_preview_widget.py
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import QTimer, pyqtSignal
import cv2
import numpy as np

from services.person_tracker import PersonTracker
from services.aws_services import AWSRekognitionService
from video_processing.video_handler import VideoHandlerMixin
from video_processing.drawing import DrawingMixin
from video_processing.calibration import CalibrationMixin
from video_processing.exporter import ExportMixin

class CCTVPreview(QWidget, VideoHandlerMixin, DrawingMixin, CalibrationMixin, ExportMixin):
    # Signals
    calibration_points_selected = pyqtSignal(list)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Shared state setup
        self.video_capture = None
        self.current_frame = None
        self.scaled_frame = None
        self.frame_offset_x = 0
        self.frame_offset_y = 0
        self.scale_factor = 1.0
        self.fps = 30
        self.frame_number = 0
        self.current_time = 0.0  # Track video time for detection intervals

        self.stores = {}
        self.cameras = {}
        self.calibration_points = []
        self.store_perspective_matrices = {}

        self.tracked_people = {}

        # Initialize person tracker with YOLO
        self.person_tracker = PersonTracker()
        # Keep AWS service for face detection only
        self.aws_service = AWSRekognitionService()

        # Initialize mixins
        self._init_video_handler()
        self._init_calibration()
        self._init_export()

        # UI setup
        self._init_ui()

    def _init_ui(self):
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip("CCTV footage preview with store detection")

    def process_frame(self, frame):
        """Process a single frame for person detection and tracking"""
        if frame is None:
            return None
        
        # Convert frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update current time
        self.current_time = self.frame_number / self.fps
        
        # Use PersonTracker's YOLO detection for all tracking
        self.tracked_people = self.person_tracker.process_frame(frame_rgb, self.stores)
        
        # Draw detections and tracking info
        frame_with_detections = self.person_tracker.draw_detections(frame_rgb)
        
        return frame_with_detections

    def draw_detections(self, frame):
        """Draw store boundaries and additional information"""
        # Draw store boundaries
        for store_id, store in self.stores.items():
            if "video_polygon" in store and len(store["video_polygon"]) > 2:
                points = np.array(store["video_polygon"], np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                
                # Draw store name
                if "name" in store:
                    centroid = np.mean(points, axis=0, dtype=np.int32)
                    cv2.putText(frame, store["name"], tuple(centroid), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face detections if available
        if self.aws_service.aws_enabled:
            for face in self.aws_service.last_face_detections:
                fx, fy, fw, fh = face['bbox']
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Blue for face
                
                # Draw face confidence
                face_label = f"Face: {face['confidence']:.1f}"
                cv2.putText(frame, face_label, (fx, fy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame
