import cv2
import numpy as np
import json
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QToolBar,
    QStatusBar, QTextEdit, QComboBox, QGroupBox, QSplitter, QDialog, QDialogButtonBox, 
    QProgressDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
import csv

from blueprint.blueprint_processor import FullBlueprintProcessor
from gui.blueprint_view import BlueprintView
from gui.cctv_preview_widget import CCTVPreview


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mall Blueprint Mapping Tool")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setSpacing(10)  # Add some spacing between widgets
        
        # Create splitter for blueprint and video views
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create main view with fixed width ratio
        self.blueprint_view = BlueprintView()
        splitter.addWidget(self.blueprint_view)
        
        # Create video preview
        self.video_preview = CCTVPreview()
        splitter.addWidget(self.video_preview)
        
        # Set initial splitter sizes (60% blueprint, 40% video)
        total_width = self.width()
        splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
        
        # Create side panel with fixed width
        side_panel = QWidget()
        side_panel.setFixedWidth(300)  # Fixed width for side panel
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(10, 10, 10, 10)  # Add some margins
        
        # Add tool selection
        tool_group = QGroupBox("Tools")
        tool_layout = QVBoxLayout(tool_group)
        tool_layout.setSpacing(5)  # Reduce spacing between tools
        
        self.select_tool_btn = QPushButton("Select")
        self.camera_tool_btn = QPushButton("Add Camera")
        self.store_tool_btn = QPushButton("Define Store")
        self.calibrate_btn = QPushButton("Calibrate Camera")
        
        # Add tooltips
        self.select_tool_btn.setToolTip("Select and move cameras or stores")
        self.camera_tool_btn.setToolTip("Click to place camera, drag to set orientation")
        self.store_tool_btn.setToolTip("Click to create store polygon, double-click to complete")
        self.calibrate_btn.setToolTip("Calibrate camera view with blueprint")
        
        # Set fixed height for buttons
        button_height = 30
        for btn in [self.select_tool_btn, self.camera_tool_btn, 
                   self.store_tool_btn, self.calibrate_btn]:
            btn.setFixedHeight(button_height)
            tool_layout.addWidget(btn)
        
        # Add export blueprint button below tools
        self.export_blueprint_btn = QPushButton("Export Blueprint")
        self.export_blueprint_btn.setFixedHeight(button_height)
        self.export_blueprint_btn.clicked.connect(self.export_data)
        tool_layout.addWidget(self.export_blueprint_btn)
        
        # Add help text with scroll area
        help_group = QGroupBox("Help")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h3>How to Use This Tool</h3>
            <ol>
                <li><b>Load Blueprint</b>
                    <ul>
                        <li>Click "Load Blueprint" in the toolbar</li>
                        <li>Select your mall blueprint image</li>
                        <li>The image should be clear and show store boundaries</li>
                    </ul>
                </li>
                <li><b>Add Cameras</b>
                    <ul>
                        <li>Click the "Add Camera" tool</li>
                        <li>Click on the blueprint where each camera is located</li>
                        <li>Drag from the camera to set its orientation</li>
                        <li>The red cone shows the camera's field of view</li>
                    </ul>
                </li>
                <li><b>Define Stores</b>
                    <ul>
                        <li>Click the "Define Store" tool</li>
                        <li>Click to create polygon points around each store</li>
                        <li>Double-click to complete the polygon</li>
                        <li>Enter store name and category when prompted</li>
                    </ul>
                </li>
                <li><b>Calibrate Camera View</b>
                    <ul>
                        <li>Load CCTV footage from a camera</li>
                        <li>Click "Calibrate Camera"</li>
                        <li>Click 4 points in the blueprint that you can identify in the video</li>
                        <li>Click the same 4 points in the video</li>
                        <li>The system will map the blueprint to the video view</li>
                    </ul>
                </li>
            </ol>
        """)
        
        help_layout.addWidget(help_text)
        
        # Add groups to side layout
        side_layout.addWidget(tool_group)
        side_layout.addWidget(help_group)
        
        # Add widgets to main layout with proper proportions
        layout.addWidget(splitter, stretch=60)  # 60% of space
        layout.addWidget(side_panel, stretch=40)  # 40% of space
        
        # Create toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)  # Prevent toolbar from being moved
        self.addToolBar(toolbar)
        
        # Add toolbar actions with tooltips
        load_blueprint_action = QAction("Load Blueprint", self)
        load_blueprint_action.setToolTip("Load a mall blueprint image")
        load_blueprint_action.triggered.connect(self.load_blueprint)
        
        load_video_action = QAction("Load CCTV", self)
        load_video_action.setToolTip("Load CCTV footage for testing")
        load_video_action.triggered.connect(self.load_video)
        
        load_mapping_action = QAction("Load Mapping", self)
        load_mapping_action.setToolTip("Load previously exported mapping data")
        load_mapping_action.triggered.connect(self.load_mapping_data)
        
        export_video_action = QAction("Export Video", self)
        export_video_action.setToolTip("Export processed video with store boundaries and person detection")
        export_video_action.triggered.connect(self.export_video)
        
        export_log_action = QAction("Export Movement Log", self)
        export_log_action.setToolTip("Export person movement log")
        export_log_action.triggered.connect(self.export_movement_log)
        
        toolbar.addAction(load_blueprint_action)
        toolbar.addAction(load_video_action)
        toolbar.addAction(load_mapping_action)
        toolbar.addAction(export_video_action)
        toolbar.addAction(export_log_action)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Connect signals
        self.select_tool_btn.clicked.connect(lambda: self.blueprint_view.set_tool("select"))
        self.camera_tool_btn.clicked.connect(lambda: self.blueprint_view.set_tool("camera"))
        self.store_tool_btn.clicked.connect(lambda: self.blueprint_view.set_tool("store"))
        self.calibrate_btn.clicked.connect(self.prepare_calibration)
        
        self.blueprint_view.calibration_points_selected.connect(self.on_blueprint_calibration_points)
        self.video_preview.calibration_points_selected.connect(self.on_video_calibration_points)
        
        # Create processor
        self.processor = FullBlueprintProcessor()
        
        # Add calibration state tracking
        self.calibration_blueprint_points = None
        self.current_calibration_store = None  # Track which store is being calibrated

    def export_data(self):
        """Export the mapping data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                # Prepare data for export
                export_data = {
                    "blueprint": {
                        "image_path": self.blueprint_view.blueprint_path if hasattr(self.blueprint_view, 'blueprint_path') else None,
                        "scale_factor": self.blueprint_view.scale_factor if hasattr(self.blueprint_view, 'scale_factor') else 1.0
                    },
                    "cameras": {
                        camera_id: {
                            "position": camera["position"],
                            "orientation": camera["orientation"],
                            "fov_angle": camera.get("fov_angle", 70),
                            "fov_range": camera.get("fov_range", 100)
                        }
                        for camera_id, camera in self.blueprint_view.cameras.items()
                    },
                    "stores": {
                        store_id: {
                            "name": store["name"],
                            "category": store.get("category", ""),
                            "polygon": store["polygon"],
                            "is_mapped": store_id in self.blueprint_view.mapped_stores
                        }
                        for store_id, store in self.blueprint_view.stores.items()
                    },
                    "calibration": {
                        "store_matrices": {
                            store_id: matrix.tolist() for store_id, matrix in self.video_preview.store_perspective_matrices.items()
                        },
                        "blueprint_points": self.calibration_blueprint_points if self.calibration_blueprint_points else None,
                        "video_points": self.video_preview.calibration_points if self.video_preview.calibration_points else None
                    },
                    "test_results": {
                        "total_stores": len(self.blueprint_view.stores),
                        "mapped_stores": len(self.blueprint_view.mapped_stores),
                        "mapping_status": {
                            store_id: {
                                "name": store["name"],
                                "is_mapped": store_id in self.blueprint_view.mapped_stores,
                                "calibration_points": self.calibration_blueprint_points if store_id == self.current_calibration_store else None
                            }
                            for store_id, store in self.blueprint_view.stores.items()
                        }
                    }
                }
                
                # Save to JSON file
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.statusBar.showMessage(f"Successfully exported data to: {file_path}")
                QMessageBox.information(self, "Export Successful", 
                    f"Mapping data has been exported to:\n{file_path}\n\n"
                    f"Exported {len(export_data['stores'])} stores and {len(export_data['cameras'])} cameras.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
                self.statusBar.showMessage("Export failed")

    def export_video(self):
        """Handle video export request"""
        if not self.video_preview.video_capture:
            QMessageBox.warning(self, "Error", "Please load a video first")
            return
        
        if not self.video_preview.stores:
            QMessageBox.warning(self, "Error", "Please define and map stores first")
            return
        
        # Enable AWS Rekognition automatically for export
        if not self.video_preview.aws_service.aws_enabled:
            if not self.video_preview.aws_service.enable_aws_rekognition(aws_region='ap-south-1'):
                QMessageBox.critical(self, "Error", "Failed to enable AWS Rekognition. Export cannot proceed.")
                return
        
        # Get output file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "Video Files (*.mp4)"
        )
        
        if file_path:
            # Ensure .mp4 extension
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            
            # Show progress dialog
            progress = QProgressDialog("Exporting video...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            
            # Connect status message to progress dialog
            def update_progress(message):
                if "Exporting video:" in message:
                    try:
                        percent = float(message.split(":")[1].strip().rstrip("%"))
                        progress.setValue(int(percent))
                    except:
                        pass
            
            self.video_preview.status_message.connect(update_progress)
            
            # Start export
            success = self.video_preview.export_video(file_path)
            
            if success:
                QMessageBox.information(self, "Success", 
                    f"Video exported successfully to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to export video")
            
            # Disconnect status message handler
            self.video_preview.status_message.disconnect(update_progress)
            
            # Reset video to first frame after export
            if self.video_preview.video_capture:
                self.video_preview.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_preview.video_capture.read()
                if ret:
                    self.video_preview.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_preview.update_scaled_frame()
                    self.video_preview.update()

    def export_movement_log(self):
        """Export the movement log of tracked people"""
        if not hasattr(self.video_preview, 'person_tracker'):
            QMessageBox.warning(self, "Error", "No movement data available")
            return
        
        # Check if we have any movement data
        has_movement_data = False
        for person in self.video_preview.person_tracker.tracked_people.values():
            if person['history']:
                has_movement_data = True
                break
        
        if not has_movement_data:
            QMessageBox.warning(self, "Error", "No movement data available. Please export a video first to generate movement data.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Movement Log", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Person ID", "Store Name", "Entry Time", "Frame Number"])
                    
                    for person_id, person in self.video_preview.person_tracker.tracked_people.items():
                        for entry in person['history']:
                            writer.writerow([
                                person_id,
                                entry['store_name'],
                                entry['entry_time'],
                                entry['frame']
                            ])
                
                QMessageBox.information(self, "Success", 
                    f"Movement log exported successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export movement log: {str(e)}")

    def load_blueprint(self):
        """Load a blueprint image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Blueprint", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            if self.blueprint_view.load_image(file_path):
                self.statusBar.showMessage(f"Loaded blueprint: {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load blueprint image")

    def load_video(self):
        """Load CCTV footage for testing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CCTV Footage", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            if self.video_preview.load_video(file_path):
                self.statusBar.showMessage(f"Loaded video: {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load video file")

    def load_mapping_data(self):
        """Load previously exported mapping data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mapping Data", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                mapping_data = json.load(f)
            
            # Debug information
            print(f"Loading mapping data from: {file_path}")
            print(f"Found {len(mapping_data.get('stores', {}))} stores")
            print(f"Found {len(mapping_data.get('cameras', {}))} cameras")
            
            # Load blueprint if path exists
            if mapping_data.get("blueprint", {}).get("image_path"):
                blueprint_path = mapping_data["blueprint"]["image_path"]
                if os.path.exists(blueprint_path):
                    if not self.blueprint_view.load_image(blueprint_path):
                        QMessageBox.warning(self, "Warning", 
                            "Could not load original blueprint image. Using current blueprint if available.")
            
            # Load stores first (needed for mapping status)
            self.blueprint_view.stores = {
                store_id: {
                    "name": store["name"],
                    "category": store.get("category", ""),
                    "polygon": store["polygon"]
                }
                for store_id, store in mapping_data.get("stores", {}).items()
            }
            
            # Update mapped stores status
            self.blueprint_view.mapped_stores = {
                store_id for store_id, store in mapping_data.get("stores", {}).items()
                if store.get("is_mapped", False)
            }
            
            # Load cameras
            self.blueprint_view.cameras = {
                camera_id: {
                    "position": tuple(camera["position"]),
                    "orientation": camera["orientation"],
                    "fov_angle": camera.get("fov_angle", 70),
                    "fov_range": camera.get("fov_range", 100)
                }
                for camera_id, camera in mapping_data.get("cameras", {}).items()
            }
            
            # Load calibration data if available
            calibration_data = mapping_data.get("calibration", {})
            if calibration_data.get("store_matrices"):
                self.video_preview.store_perspective_matrices = {
                    store_id: np.array(matrix, dtype=np.float32) for store_id, matrix in calibration_data["store_matrices"].items()
                }
                print("Loaded perspective matrices for stores")
            else:
                print("No perspective matrices found in mapping data")
            
            # Copy stores to video preview
            self.video_preview.stores = self.blueprint_view.stores.copy()
            
            # Update UI
            self.blueprint_view.update()
            self.video_preview.update()
            self.statusBar.showMessage(f"Loaded mapping data from: {file_path}")
            
            # Show summary
            QMessageBox.information(self, "Mapping Data Loaded",
                f"Successfully loaded mapping data:\n\n"
                f"• {len(self.blueprint_view.stores)} stores\n"
                f"• {len(self.blueprint_view.cameras)} cameras\n"
                f"• {len(self.blueprint_view.mapped_stores)} mapped stores\n"
                f"• Perspective matrices: {len(self.video_preview.store_perspective_matrices)} stores\n"
                f"• Matrix shapes: {[matrix.shape for matrix in self.video_preview.store_perspective_matrices.values()]}\n\n"
                "You can now load a new video to test the mapping.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mapping data: {str(e)}")
            self.statusBar.showMessage("Failed to load mapping data")
            print(f"Error loading mapping data: {str(e)}")

    def prepare_calibration(self):
        """Prepare for calibration by selecting a store"""
        if self.blueprint_view.blueprint_image is None or self.video_preview.current_frame is None:
            QMessageBox.warning(self, "Error", "Please load both blueprint and video first")
            return
        
        # Check if there are any stores defined
        if not self.blueprint_view.stores:
            QMessageBox.warning(self, "Error", "Please define at least one store before calibration")
            return
        
        # Create store selection dialog
        store_dialog = QDialog(self)
        store_dialog.setWindowTitle("Select Store for Calibration")
        store_dialog.setModal(True)
        
        layout = QVBoxLayout(store_dialog)
        
        # Add store selection combo box
        store_combo = QComboBox()
        for store_id, store in self.blueprint_view.stores.items():
            store_combo.addItem(f"{store['name']} ({store_id})", store_id)
        layout.addWidget(QLabel("Select store to calibrate:"))
        layout.addWidget(store_combo)
        
        # Add calibration instructions
        instructions = QLabel(
            "IMPORTANT: The blueprint shows stores vertically, but in the video they appear horizontally.\n\n"
            "When selecting points:\n"
            "1. In the blueprint, click points in this order:\n"
            "   • Top-left corner of the store area\n"
            "   • Top-right corner of the store area\n"
            "   • Bottom-right corner of the store area\n"
            "   • Bottom-left corner of the store area\n"
            "2. In the video, click the same 4 points in the same order\n"
            "   (The points should form a rectangle around the stores)"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(store_dialog.accept)
        button_box.rejected.connect(store_dialog.reject)
        layout.addWidget(button_box)
        
        if store_dialog.exec() == QDialog.DialogCode.Accepted:
            selected_store_id = store_combo.currentData()
            self.current_calibration_store = selected_store_id
            
            # Show calibration instructions
            dialog = CalibrationDialog(self)
            dialog.setText(f"Calibrating Camera for {self.blueprint_view.stores[selected_store_id]['name']}")
            
            if dialog.exec() == QMessageBox.StandardButton.Ok:
                self.blueprint_view.set_tool("calibrate")
                self.video_preview.set_calibration_mode(True)
                self.statusBar.showMessage(
                    f"Step 1: Click 4 points around {self.blueprint_view.stores[selected_store_id]['name']} "
                    "in the blueprint (top-left, top-right, bottom-right, bottom-left)"
                )

    def on_blueprint_calibration_points(self, points):
        """Handle blueprint calibration points selection"""
        self.calibration_blueprint_points = points
        self.statusBar.showMessage(
            "Step 2: Now click the same 4 points in the video in the same order "
            "(top-left, top-right, bottom-right, bottom-left)"
        )

    def on_video_calibration_points(self, points):
        """Handle video calibration points selection"""
        if self.calibration_blueprint_points is None or self.current_calibration_store is None:
            return
        
        # Calculate perspective transform for this specific store
        if self.video_preview.calculate_perspective_transform(
            self.calibration_blueprint_points, points, self.current_calibration_store):
            
            # Update mapping status for the current store only
            self.blueprint_view.mapped_stores.add(self.current_calibration_store)
            
            store_name = self.blueprint_view.stores[self.current_calibration_store]['name']
            self.statusBar.showMessage(f"Camera calibration complete - {store_name} mapped")
            
            # Show transformation details
            QMessageBox.information(self, "Calibration Complete", 
                f"Camera calibration successful!\n\n"
                f"Store '{store_name}' has been mapped to the video view.\n\n"
                f"Each store now has its own perspective transformation, ensuring correct positioning "
                f"relative to other stores in the CCTV view.\n\n"
                f"You can now test the mapping to verify the store boundaries are correctly positioned.")
        else:
            QMessageBox.warning(self, "Error", 
                "Calibration failed. Please try again, making sure to:\n\n"
                "1. Click points in the correct order (top-left, top-right, bottom-right, bottom-left)\n"
                "2. Select points that form a proper rectangle around the stores\n"
                "3. Ensure the points in the video correspond to the same locations as in the blueprint")
        
        # Reset calibration mode
        self.blueprint_view.set_tool("select")
        self.video_preview.set_calibration_mode(False)
        self.calibration_blueprint_points = None
        self.current_calibration_store = None
        self.blueprint_view.update()


class CalibrationDialog(QMessageBox):
    """Dialog for camera calibration"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration")
        self.setText("Camera Calibration Required")
        self.setInformativeText(
            "To map the blueprint to CCTV footage, we need to calibrate the camera.\n\n"
            "IMPORTANT: The blueprint shows stores vertically, but in the video they appear horizontally.\n\n"
            "Calibration Steps:\n"
            "1. In the blueprint, click 4 points in this order:\n"
            "   • Top-left corner of the store area\n"
            "   • Top-right corner of the store area\n"
            "   • Bottom-right corner of the store area\n"
            "   • Bottom-left corner of the store area\n"
            "2. In the video, click the same 4 points in the same order\n"
            "   (The points should form a rectangle around the stores)\n"
            "3. The system will calculate the correct perspective transformation"
        )
        self.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        self.setDefaultButton(QMessageBox.StandardButton.Ok)