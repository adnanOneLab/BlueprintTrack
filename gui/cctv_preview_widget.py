# cctv_preview_widget.py
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import QTimer, pyqtSignal

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

        self.stores = {}
        self.cameras = {}
        self.calibration_points = []
        self.store_perspective_matrices = {}

        self.tracked_people = {}

        self.person_tracker = PersonTracker()
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
