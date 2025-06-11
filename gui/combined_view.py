from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter, QTabWidget
from PyQt6.QtCore import Qt, pyqtSignal
from gui.blueprint_view import BlueprintView
from gui.cctv_preview_widget import CCTVPreview
import os

class CombinedView(QWidget):
    """Widget that combines blueprint and CCTV views in a single tab"""
    # Signals
    calibration_points_selected = pyqtSignal(list)  # Forward calibration points to main window
    status_message = pyqtSignal(str)  # Forward status messages
    tab_title_changed = pyqtSignal(str)  # Signal to update tab title
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create splitter for resizable views
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create blueprint and CCTV views
        self.blueprint_view = BlueprintView()
        self.video_preview = CCTVPreview()
        
        # Add views to splitter
        splitter.addWidget(self.blueprint_view)
        splitter.addWidget(self.video_preview)
        
        # Set initial splitter sizes (equal width)
        splitter.setSizes([self.width() // 2, self.width() // 2])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        # Connect signals
        self.blueprint_view.calibration_points_selected.connect(self.on_blueprint_calibration_points)
        self.video_preview.calibration_points_selected.connect(self.on_video_calibration_points)
        self.video_preview.status_message.connect(self.status_message)
        
        # Connect blueprint view signals to update status
        self.blueprint_view.image_loaded.connect(self.on_blueprint_loaded)
        self.blueprint_view.camera_added.connect(self.on_blueprint_updated)
        self.blueprint_view.store_added.connect(self.on_blueprint_updated)
        
    def on_blueprint_calibration_points(self, points):
        """Handle blueprint calibration points and forward to video preview"""
        # Store points temporarily and forward to main window
        self.calibration_points_selected.emit(points)
        
    def on_video_calibration_points(self, points):
        """Handle video calibration points"""
        # Forward to main window
        self.calibration_points_selected.emit(points)
        
    def on_blueprint_loaded(self):
        """Handle blueprint image loaded"""
        # Update tab title through signal
        self.tab_title_changed.emit(self.get_tab_title())
            
    def on_blueprint_updated(self):
        """Handle blueprint updates (cameras/stores added)"""
        # Update tab title through signal
        self.tab_title_changed.emit(self.get_tab_title())
        
    def load_blueprint(self, image_path):
        """Load blueprint image"""
        success = self.blueprint_view.load_image(image_path)
        if success:
            # Copy stores to video preview
            self.video_preview.stores = self.blueprint_view.stores.copy()
            # Update tab title
            self.tab_title_changed.emit(self.get_tab_title())
        return success
        
    def load_video(self, video_path):
        """Load CCTV video"""
        success = self.video_preview.load_video(video_path)
        if success:
            # Update tab title
            self.tab_title_changed.emit(self.get_tab_title())
        return success
        
    def get_tab_title(self):
        """Get a title for this tab based on loaded content"""
        blueprint_name = os.path.basename(self.blueprint_view.blueprint_path) if hasattr(self.blueprint_view, 'blueprint_path') else "No Blueprint"
        video_name = os.path.basename(self.video_preview.video_path) if hasattr(self.video_preview, 'video_path') else "No Video"
        
        # Get location from PersonTracker if available
        location = "Unknown"
        if hasattr(self.video_preview, 'person_tracker'):
            location = self.video_preview.person_tracker.location
        
        return f"[{location}] {blueprint_name} - {video_name}"
        
    def cleanup(self):
        """Clean up resources when tab is closed"""
        # Stop video playback
        if hasattr(self.video_preview, 'timer') and self.video_preview.timer.isActive():
            self.video_preview.timer.stop()
            
        if self.video_preview.video_capture:
            self.video_preview.video_capture.release()
            self.video_preview.video_capture = None
            
        # Clear any stored data
        self.video_preview.current_frame = None
        self.video_preview.scaled_frame = None
        self.video_preview.stores = {}
        self.video_preview.cameras = {}
        self.video_preview.calibration_points = []
        self.video_preview.store_perspective_matrices = {}
        
        # Clear blueprint data
        self.blueprint_view.blueprint_image = None
        self.blueprint_view.scaled_image = None
        self.blueprint_view.stores = {}
        self.blueprint_view.cameras = {}
        self.blueprint_view.mapped_stores = set()
        
    def showEvent(self, event):
        """Handle tab becoming visible"""
        super().showEvent(event)
        # Ensure video preview is updated when tab becomes visible
        if self.video_preview.current_frame is not None:
            self.video_preview.update()
            
    def hideEvent(self, event):
        """Handle tab becoming hidden"""
        super().hideEvent(event)
        # Pause video playback when tab is hidden
        if hasattr(self.video_preview, 'timer') and self.video_preview.timer.isActive():
            self.video_preview.timer.stop()
            
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        # Update splitter sizes to maintain equal width
        if hasattr(self, 'splitter'):
            self.splitter.setSizes([self.width() // 2, self.width() // 2]) 