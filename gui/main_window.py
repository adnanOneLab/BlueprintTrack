# Standard library imports
import csv
import json
import os

# Third-party imports
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QFont, QPainter, QImage
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QToolBar,
    QStatusBar, QTextEdit, QComboBox, QGroupBox, QSplitter, QDialog, 
    QDialogButtonBox, QProgressDialog, QTabWidget, QScrollArea, QSizePolicy
)

# Local imports
from blueprint.blueprint_processor import FullBlueprintProcessor
from gui.blueprint_view import BlueprintView
from gui.cctv_preview_widget import CCTVPreview


class MainWindow(QMainWindow):
    """Main application window with enhanced GUI"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mall Blueprint Mapping Tool")
        self.setMinimumSize(1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #aaa;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QToolBar {
                background: #e0e0e0;
                border: none;
                padding: 2px;
            }
            QStatusBar {
                background: #e0e0e0;
            }
            QTextEdit {
                background: white;
                border: 1px solid #ccc;
            }
            QComboBox {
                padding: 3px;
                border: 1px solid #aaa;
                border-radius: 3px;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create left panel with blueprint and video
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setSpacing(5)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget for blueprint and video
        self.view_tabs = QTabWidget()
        self.view_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setTabsClosable(False)
        
        # Create blueprint view
        self.blueprint_view = BlueprintView()
        self.view_tabs.addTab(self.blueprint_view, "Blueprint View")
        
        # Create video preview
        self.video_preview = CCTVPreview()
        self.view_tabs.addTab(self.video_preview, "CCTV View")
        
        left_panel_layout.addWidget(self.view_tabs)
        
        # Create right panel with controls
        right_panel = QWidget()
        right_panel.setFixedWidth(350)
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setSpacing(10)
        right_panel_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add logo/header
        header = QLabel("Mall Blueprint Mapping Tool")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(14)
        header.setFont(header_font)
        right_panel_layout.addWidget(header)
        
        # Create tab widget for right panel
        control_tabs = QTabWidget()
        control_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Tools tab
        tools_tab = QWidget()
        tools_layout = QVBoxLayout(tools_tab)
        
        # Tools group
        tools_group = QGroupBox("Mapping Tools")
        tools_group_layout = QVBoxLayout(tools_group)
        
        # Create tool buttons with icons
        self.select_tool_btn = self.create_tool_button("Select", "select.svg")
        self.camera_tool_btn = self.create_tool_button("Add Camera", "camera.svg")
        self.store_tool_btn = self.create_tool_button("Define Store", "store.svg")
        self.calibrate_btn = self.create_tool_button("Calibrate Camera", "calibrate.svg")
        
        # Add tooltips
        self.select_tool_btn.setToolTip("Select and move cameras or stores")
        self.camera_tool_btn.setToolTip("Click to place camera, drag to set orientation")
        self.store_tool_btn.setToolTip("Click to create store polygon, double-click to complete")
        self.calibrate_btn.setToolTip("Calibrate camera view with blueprint")
        
        # Add buttons to layout
        tools_group_layout.addWidget(self.select_tool_btn)
        tools_group_layout.addWidget(self.camera_tool_btn)
        tools_group_layout.addWidget(self.store_tool_btn)
        tools_group_layout.addWidget(self.calibrate_btn)
        
        # Export button
        self.export_blueprint_btn = self.create_tool_button("Export Blueprint", "export.svg")
        self.export_blueprint_btn.clicked.connect(self.export_data)
        tools_group_layout.addWidget(self.export_blueprint_btn)
        
        tools_layout.addWidget(tools_group)
        
        # Status group
        status_group = QGroupBox("Mapping Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("No blueprint loaded")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        self.camera_count_label = QLabel("Cameras: 0")
        status_layout.addWidget(self.camera_count_label)
        
        self.store_count_label = QLabel("Stores: 0 (0 mapped)")
        status_layout.addWidget(self.store_count_label)
        
        tools_layout.addWidget(status_group)
        tools_layout.addStretch()
        
        # Help tab
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        
        # Create scroll area for help content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        help_content = QWidget()
        help_content_layout = QVBoxLayout(help_content)
        
        # Help text with improved formatting
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
            <h2 style="color: #2a5caa;">Mall Blueprint Mapping Tool</h2>
            <h3>How to Use This Tool</h3>
            <ol>
                <li><b style="color: #2a5caa;">Load Blueprint</b>
                    <ul>
                        <li>Click "Load Blueprint" in the toolbar</li>
                        <li>Select your mall blueprint image</li>
                        <li>The image should be clear and show store boundaries</li>
                    </ul>
                </li>
                <li><b style="color: #2a5caa;">Add Cameras</b>
                    <ul>
                        <li>Click the "Add Camera" tool</li>
                        <li>Click on the blueprint where each camera is located</li>
                        <li>Drag from the camera to set its orientation</li>
                        <li>The red cone shows the camera's field of view</li>
                    </ul>
                </li>
                <li><b style="color: #2a5caa;">Define Stores</b>
                    <ul>
                        <li>Click the "Define Store" tool</li>
                        <li>Click to create polygon points around each store</li>
                        <li>Double-click to complete the polygon</li>
                        <li>Enter store name and category when prompted</li>
                    </ul>
                </li>
                <li><b style="color: #2a5caa;">Calibrate Camera View</b>
                    <ul>
                        <li>Load CCTV footage from a camera</li>
                        <li>Click "Calibrate Camera"</li>
                        <li>Click 4 points in the blueprint that you can identify in the video</li>
                        <li>Click the same 4 points in the video</li>
                        <li>The system will map the blueprint to the video view</li>
                    </ul>
                </li>
            </ol>
            <h3>Tips for Best Results</h3>
            <ul>
                <li>Use high-resolution blueprint images for better accuracy</li>
                <li>When calibrating, choose points that are clearly visible in both views</li>
                <li>For stores, include the entire retail space in your polygon</li>
                <li>Save your work frequently using the Export Blueprint option</li>
            </ul>
        """)
        
        help_content_layout.addWidget(help_text)
        scroll_area.setWidget(help_content)
        help_layout.addWidget(scroll_area)
        
        # Add tabs to control panel
        control_tabs.addTab(tools_tab, "Tools")
        control_tabs.addTab(help_tab, "Help")
        
        right_panel_layout.addWidget(control_tabs)
        
        # Add panels to main splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        main_splitter.setSizes([self.width() - 350, 350])
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
        # Create enhanced toolbar
        self.create_toolbar()
        
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
        
        # Connect blueprint view signals to update status
        self.blueprint_view.image_loaded.connect(self.update_status)
        self.blueprint_view.camera_added.connect(self.update_status)
        self.blueprint_view.store_added.connect(self.update_status)
        
        # Create processor
        self.processor = FullBlueprintProcessor()
        
        # Add calibration state tracking
        self.calibration_blueprint_points = None
        self.current_calibration_store = None
        
        # Update initial status
        self.update_status()

    def create_tool_button(self, text, icon_name=None):
        """Create a styled tool button with optional icon"""
        button = QPushButton(text)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setMinimumHeight(40)
        
        if icon_name:
            # Try to load icon (fallback to text if not found)
            icon_path = os.path.join("assets", "icons", icon_name)
            if os.path.exists(icon_path):
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QSize(24, 24))
        
        return button

    def create_toolbar(self):
        """Create enhanced toolbar with icons and tooltips"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Create actions with icons
        actions = [
            ("Load Blueprint", "blueprint.svg", "Load a mall blueprint image", self.load_blueprint),
            ("Load CCTV", "cctv.svg", "Load CCTV footage for testing", self.load_video),
            ("Load Mapping", "import.svg", "Load previously exported mapping data", self.load_mapping_data),
            ("Export Video", "video_export.svg", "Export processed video with store boundaries and person detection", self.export_video),
            ("Export Log", "log_export.svg", "Export person movement log", self.export_movement_log),
        ]
        
        for text, icon, tooltip, callback in actions:
            action = QAction(text, self)
            action.setToolTip(tooltip)
            icon_path = os.path.join("assets", "icons", icon)
            if os.path.exists(icon_path):
                action.setIcon(QIcon(icon_path))
            action.triggered.connect(callback)
            toolbar.addAction(action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Add quick help button
        help_action = QAction("Quick Help", self)
        help_action.setToolTip("Show quick help reference")
        help_action.setIcon(QIcon.fromTheme("help-contents"))
        help_action.triggered.connect(self.show_quick_help)
        toolbar.addAction(help_action)

    def update_status(self):
        """Update the status labels based on current state"""
        if self.blueprint_view.blueprint_image is not None:
            self.status_label.setText(f"Blueprint loaded: {os.path.basename(self.blueprint_view.blueprint_path)}")
        else:
            self.status_label.setText("No blueprint loaded")
        
        camera_count = len(self.blueprint_view.cameras)
        self.camera_count_label.setText(f"Cameras: {camera_count}")
        
        store_count = len(self.blueprint_view.stores)
        mapped_count = len(self.blueprint_view.mapped_stores)
        self.store_count_label.setText(f"Stores: {store_count} ({mapped_count} mapped)")

    def show_quick_help(self):
        """Show a quick help dialog"""
        help_dialog = QMessageBox(self)
        help_dialog.setWindowTitle("Quick Help")
        help_dialog.setIcon(QMessageBox.Icon.Information)
        help_dialog.setTextFormat(Qt.TextFormat.RichText)
        help_dialog.setText("""
            <h3>Keyboard Shortcuts</h3>
            <ul>
                <li><b>Ctrl+O</b>: Load blueprint</li>
                <li><b>Ctrl+V</b>: Load CCTV video</li>
                <li><b>Ctrl+S</b>: Save mapping data</li>
                <li><b>Esc</b>: Cancel current tool</li>
                <li><b>Delete</b>: Remove selected item</li>
            </ul>
            <h3>Tool Tips</h3>
            <ul>
                <li>Right-click on items for context menu</li>
                <li>Double-click stores to edit properties</li>
                <li>Drag cameras to adjust their position</li>
                <li>Drag camera direction line to adjust view angle</li>
            </ul>
        """)
        help_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        help_dialog.exec()

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
        default_folder = os.path.join(os.getcwd(), "exports/output/")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", default_folder, "Video Files (*.mp4)"
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
        
        default_folder = os.path.join(os.getcwd(), "exports/logs/")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Movement Log", default_folder, "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Person ID", "Store Name", "Entry Time", "Frame Number", "Duration (frames)"])
                    
                    # Process each person's history
                    for person_id, person in self.video_preview.person_tracker.tracked_people.items():
                        if not person['history']:
                            continue
                            
                        # Sort history by frame number to ensure chronological order
                        sorted_history = sorted(person['history'], key=lambda x: x['frame'])
                        
                        # Track unique store visits
                        current_store = None
                        entry_frame = None
                        entry_time = None
                        
                        for i, entry in enumerate(sorted_history):
                            # If this is a new store or the first entry
                            if entry['store_name'] != current_store:
                                # If we were tracking a previous store, write its entry
                                if current_store is not None and entry_frame is not None:
                                    duration = entry['frame'] - entry_frame
                                    writer.writerow([
                                        person_id,
                                        current_store,
                                        entry_time,
                                        entry_frame,
                                        duration
                                    ])
                                
                                # Start tracking new store
                                current_store = entry['store_name']
                                entry_frame = entry['frame']
                                entry_time = entry['entry_time']
                            
                            # If this is the last entry, write it
                            if i == len(sorted_history) - 1:
                                duration = entry['frame'] - entry_frame
                                writer.writerow([
                                    person_id,
                                    current_store,
                                    entry_time,
                                    entry_frame,
                                    duration
                                ])
                
                QMessageBox.information(self, "Success", 
                    f"Movement log exported successfully to:\n{file_path}\n\n"
                    "The log includes:\n"
                    "• One entry per store visit\n"
                    "• Entry time and frame number\n"
                    "• Duration of stay in frames")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export movement log: {str(e)}")

    def load_blueprint(self):
        """Load a blueprint image"""
        default_folder = os.path.join(os.getcwd(), "assets/blueprints/")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Blueprint", default_folder, "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            if self.blueprint_view.load_image(file_path):
                self.statusBar.showMessage(f"Loaded blueprint: {file_path}")
                self.update_status()
            else:
                QMessageBox.critical(self, "Error", "Failed to load blueprint image")

    def load_video(self):
        """Load CCTV footage for testing"""
        default_folder = os.path.join(os.getcwd(), "assets/input_videos/")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CCTV Footage", default_folder, "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            if self.video_preview.load_video(file_path):
                self.statusBar.showMessage(f"Loaded video: {file_path}")
                # Switch to video tab
                self.view_tabs.setCurrentIndex(1)
            else:
                QMessageBox.critical(self, "Error", "Failed to load video file")

    def load_mapping_data(self):
        """Load previously exported mapping data"""
        default_folder = os.path.join(os.getcwd(), "assets/mappings/")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mapping Data", default_folder, "JSON Files (*.json)"
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
            self.update_status()
            
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
        if not points or len(points) != 4:
            QMessageBox.warning(self, "Error", "Please select exactly 4 points for calibration")
            return
            
        # Verify points form a reasonable quadrilateral
        points_array = np.array(points)
        if not self._verify_calibration_points(points_array):
            QMessageBox.warning(self, "Error", 
                "Selected points do not form a valid quadrilateral.\n"
                "Please select points that form a proper rectangle around the store.")
            return
            
        self.calibration_blueprint_points = points
        self.statusBar.showMessage(
            "Step 2: Now click the same 4 points in the video in the same order "
            "(top-left, top-right, bottom-right, bottom-left)"
        )

    def on_video_calibration_points(self, points):
        """Handle video calibration points selection"""
        if not points or len(points) != 4:
            QMessageBox.warning(self, "Error", "Please select exactly 4 points for calibration")
            return
            
        if self.calibration_blueprint_points is None or self.current_calibration_store is None:
            QMessageBox.warning(self, "Error", "Blueprint calibration points not set. Please start over.")
            return
            
        # Verify points form a reasonable quadrilateral
        points_array = np.array(points)
        if not self._verify_calibration_points(points_array):
            QMessageBox.warning(self, "Error", 
                "Selected points do not form a valid quadrilateral.\n"
                "Please select points that form a proper rectangle around the store.")
            return
        
        try:
            # Calculate perspective transform for this specific store
            if self.video_preview.calculate_perspective_transform(
                self.calibration_blueprint_points, points, self.current_calibration_store):
                
                # Update mapping status for the current store only
                self.blueprint_view.mapped_stores.add(self.current_calibration_store)
                
                store_name = self.blueprint_view.stores[self.current_calibration_store]['name']
                self.statusBar.showMessage(f"Camera calibration complete - {store_name} mapped")
                self.update_status()
                
                # Show transformation details
                QMessageBox.information(self, "Calibration Complete", 
                    f"Camera calibration successful!\n\n"
                    f"Store '{store_name}' has been mapped to the video view.\n\n"
                    f"Each store now has its own perspective transformation, ensuring correct positioning "
                    f"relative to other stores in the CCTV view.\n\n"
                    f"You can now test the mapping to verify the store boundaries are correctly positioned.")
            else:
                raise ValueError("Failed to calculate perspective transform")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                f"Calibration failed: {str(e)}\n\n"
                "Please try again, making sure to:\n"
                "1. Click points in the correct order (top-left, top-right, bottom-right, bottom-left)\n"
                "2. Select points that form a proper rectangle around the stores\n"
                "3. Ensure the points in the video correspond to the same locations as in the blueprint")
        finally:
            # Reset calibration mode
            self.blueprint_view.set_tool("select")
            self.video_preview.set_calibration_mode(False)
            self.calibration_blueprint_points = None
            self.current_calibration_store = None
            self.blueprint_view.update()

    def _verify_calibration_points(self, points):
        """Verify that the calibration points form a reasonable quadrilateral"""
        if len(points) != 4:
            return False
            
        # Convert to numpy array if not already
        points = np.array(points)
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(4):
            j = (i + 1) % 4
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
            
        # Check if any side is too short (less than 10 pixels)
        if min(distances) < 10:
            return False
            
        # Check if the quadrilateral is too skewed
        # Calculate angles between consecutive sides
        angles = []
        for i in range(4):
            j = (i + 1) % 4
            k = (i + 2) % 4
            v1 = points[j] - points[i]
            v2 = points[k] - points[j]
            # Normalize vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            # Calculate angle
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
            
        # Check if any angle is too acute (less than 30 degrees) or too obtuse (more than 150 degrees)
        if min(angles) < 30 or max(angles) > 150:
            return False
            
        return True


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