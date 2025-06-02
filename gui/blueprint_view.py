from PyQt6.QtWidgets import QWidget, QDialog, QDialogButtonBox, QLineEdit, QFormLayout, QMessageBox, QSizePolicy
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QFont
import numpy as np
import cv2

class BlueprintView(QWidget):
    camera_placed = pyqtSignal(str, int, int, float)  # camera_id, x, y, orientation
    store_defined = pyqtSignal(str, list)  # store_id, polygon_points
    calibration_points_selected = pyqtSignal(list)  # List of (x, y) points

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blueprint_image = None
        self.blueprint_path = None  # Add path storage
        self.scaled_image = None
        self.current_tool = "select"  # select, camera, store, calibrate
        self.drawing_points = []
        self.cameras = {}
        self.stores = {}
        self.calibration_points = []
        self.setMouseTracking(True)
        self.setToolTip("Click to place cameras or define store boundaries")
        
        # Add store mapping status
        self.mapped_stores = set()  # Track which stores have been mapped
        
        # Set minimum size to ensure the widget is visible
        self.setMinimumSize(400, 300)
        
        # Set size policy to allow widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def set_tool(self, tool):
        """Set the current interaction tool"""
        self.current_tool = tool
        self.drawing_points = []
        if tool == "calibrate":
            self.calibration_points = []
            self.setToolTip("Click 4 points for camera calibration")
        elif tool == "camera":
            self.setToolTip("Click to place camera, drag to set orientation")
        elif tool == "store":
            self.setToolTip("Click to add polygon points, double-click to complete")
        else:
            self.setToolTip("Click to select and move elements")
        self.update()
    
    def load_image(self, image_path):
        """Load and display the blueprint image"""
        self.blueprint_image = cv2.imread(image_path)
        if self.blueprint_image is None:
            return False
        
        # Store the blueprint path
        self.blueprint_path = image_path
        
        # Convert to RGB for display
        self.blueprint_image = cv2.cvtColor(self.blueprint_image, cv2.COLOR_BGR2RGB)
        
        # Update the scaled image
        self.update_scaled_image()
        
        # Update the widget size to match the image
        self.updateGeometry()
        
        return True
    
    def update_scaled_image(self):
        """Update the scaled image to fit the widget size"""
        if self.blueprint_image is None:
            return
        
        # Get the widget size
        widget_size = self.size()
        
        # Calculate the scaling factor to fit the image in the widget
        img_height, img_width = self.blueprint_image.shape[:2]
        scale_w = widget_size.width() / img_width
        scale_h = widget_size.height() / img_height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        self.scaled_image = cv2.resize(self.blueprint_image, (new_width, new_height))
        
        # Calculate the offset to center the image
        self.image_offset_x = (widget_size.width() - new_width) // 2
        self.image_offset_y = (widget_size.height() - new_height) // 2
        
        # Store the scale factor for coordinate conversion
        self.scale_factor = scale
    
    def resizeEvent(self, event):
        """Handle widget resize events"""
        super().resizeEvent(event)
        self.update_scaled_image()
        self.update()
    
    def paintEvent(self, event):
        """Draw the blueprint and any overlays"""
        if self.scaled_image is None:
            return
        
        painter = QPainter(self)
        
        # Draw blueprint
        height, width = self.scaled_image.shape[:2]
        qimage = QImage(self.scaled_image.data, width, height, 
                       self.scaled_image.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(self.image_offset_x, self.image_offset_y, qimage)
        
        # Convert coordinates for overlays
        def to_widget_coords(x, y):
            return (int(x * self.scale_factor + self.image_offset_x),
                   int(y * self.scale_factor + self.image_offset_y))
        
        # Draw cameras
        for camera_id, camera in self.cameras.items():
            x, y = to_widget_coords(*camera["position"])
            orientation = camera["orientation"]
            
            # Draw camera marker
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawEllipse(QPoint(x, y), 5, 5)
            
            # Draw orientation line
            line_length = 20 * self.scale_factor
            end_x = int(x + line_length * np.cos(np.radians(orientation)))
            end_y = int(y - line_length * np.sin(np.radians(orientation)))
            painter.drawLine(x, y, end_x, end_y)
            
            # Draw camera ID
            painter.setFont(QFont("Arial", 8))
            painter.drawText(x + 10, y, camera_id)
            
            # Draw field of view cone
            if "fov_angle" in camera and "fov_range" in camera:
                fov_angle = camera["fov_angle"]
                fov_range = camera["fov_range"] * self.scale_factor
                half_angle = fov_angle / 2
                
                # Draw cone edges
                angle1 = np.radians(orientation - half_angle)
                angle2 = np.radians(orientation + half_angle)
                
                edge1_x = int(x + fov_range * np.cos(angle1))
                edge1_y = int(y - fov_range * np.sin(angle1))
                edge2_x = int(x + fov_range * np.cos(angle2))
                edge2_y = int(y - fov_range * np.sin(angle2))
                
                painter.setPen(QPen(QColor(255, 0, 0), 1, Qt.PenStyle.DashLine))
                painter.drawLine(x, y, edge1_x, edge1_y)
                painter.drawLine(x, y, edge2_x, edge2_y)
        
        # Draw stores
        for store_id, store in self.stores.items():
            points = store["polygon"]
            if len(points) > 1:
                # Convert points to widget coordinates
                widget_points = [to_widget_coords(x, y) for x, y in points]
                
                # Set color based on mapping status
                if store_id in self.mapped_stores:
                    painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green for mapped stores
                else:
                    painter.setPen(QPen(QColor(255, 165, 0), 2))  # Orange for unmapped stores
                
                # Draw polygon
                for i in range(len(widget_points) - 1):
                    painter.drawLine(widget_points[i][0], widget_points[i][1],
                                   widget_points[i+1][0], widget_points[i+1][1])
                if len(widget_points) > 2:
                    painter.drawLine(widget_points[-1][0], widget_points[-1][1],
                                   widget_points[0][0], widget_points[0][1])
                
                # Draw store name and mapping status
                if "name" in store:
                    centroid_x = int(np.mean([p[0] for p in widget_points]))
                    centroid_y = int(np.mean([p[1] for p in widget_points]))
                    painter.setFont(QFont("Arial", 8))
                    status = "✓" if store_id in self.mapped_stores else "?"
                    painter.drawText(centroid_x, centroid_y, f"{store['name']} {status}")
        
        # Draw current drawing points
        if self.current_tool == "store" and len(self.drawing_points) > 0:
            # Convert points to widget coordinates
            widget_points = [to_widget_coords(x, y) for x, y in self.drawing_points]
            
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            for i in range(len(widget_points) - 1):
                painter.drawLine(widget_points[i][0], widget_points[i][1],
                               widget_points[i+1][0], widget_points[i+1][1])
            if len(widget_points) > 2:
                painter.drawLine(widget_points[-1][0], widget_points[-1][1],
                               widget_points[0][0], widget_points[0][1])
        
        # Draw calibration points
        if self.current_tool == "calibrate":
            for i, point in enumerate(self.calibration_points):
                x, y = to_widget_coords(*point)
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.drawEllipse(QPoint(int(x), int(y)), 5, 5)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(int(x + 10), int(y), str(i + 1))
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for different tools"""
        if self.scaled_image is None:
            return
        
        # Convert mouse coordinates to image coordinates
        x = (event.position().x() - self.image_offset_x) / self.scale_factor
        y = (event.position().y() - self.image_offset_y) / self.scale_factor
        
        # Check if click is within image bounds
        img_height, img_width = self.blueprint_image.shape[:2]
        if not (0 <= x < img_width and 0 <= y < img_height):
            return
        
        if self.current_tool == "camera":
            # Place a new camera
            camera_id = f"cam{len(self.cameras) + 1:03d}"
            self.cameras[camera_id] = {
                "position": (x, y),
                "orientation": 0,
                "fov_angle": 70,  # Default FOV
                "fov_range": 100  # Default range in pixels
            }
            self.camera_placed.emit(camera_id, x, y, 0)
            self.update()
        
        elif self.current_tool == "store":
            # Add point to store polygon
            self.drawing_points.append((x, y))
            self.update()
        
        elif self.current_tool == "calibrate":
            # Add calibration point
            if len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                if len(self.calibration_points) == 4:
                    self.calibration_points_selected.emit(self.calibration_points)
                self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera orientation"""
        if self.current_tool == "camera" and event.buttons() & Qt.MouseButton.LeftButton:
            # Convert mouse coordinates to image coordinates
            mouse_x = (event.position().x() - self.image_offset_x) / self.scale_factor
            mouse_y = (event.position().y() - self.image_offset_y) / self.scale_factor
            
            # Update camera orientation
            for camera_id, camera in self.cameras.items():
                cam_x, cam_y = camera["position"]
                dx = mouse_x - cam_x
                dy = cam_y - mouse_y  # Invert y-axis
                orientation = np.degrees(np.arctan2(dy, dx))
                camera["orientation"] = orientation
                self.camera_placed.emit(camera_id, cam_x, cam_y, orientation)
                self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Handle double clicks for completing store polygons"""
        if self.current_tool == "store" and len(self.drawing_points) > 2:
            # Create store dialog
            store_dialog = QDialog(self)
            store_dialog.setWindowTitle("Store Information")
            store_dialog.setModal(True)
            
            # Create form layout
            form_layout = QFormLayout(store_dialog)
            
            # Add input fields
            name_input = QLineEdit()
            category_input = QLineEdit()
            form_layout.addRow("Store Name:", name_input)
            form_layout.addRow("Category:", category_input)
            
            # Add buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(store_dialog.accept)
            button_box.rejected.connect(store_dialog.reject)
            form_layout.addRow(button_box)
            
            # Show dialog
            if store_dialog.exec() == QDialog.DialogCode.Accepted:
                store_name = name_input.text().strip()
                store_category = category_input.text().strip()
                
                if not store_name:
                    QMessageBox.warning(self, "Error", "Store name cannot be empty")
                    return
                
                # Generate store ID
                store_id = f"store{len(self.stores) + 1:03d}"
                
                # Add store
                self.stores[store_id] = {
                    "polygon": self.drawing_points.copy(),
                    "name": store_name,
                    "category": store_category
                }
                self.store_defined.emit(store_id, self.drawing_points)
                self.drawing_points = []
                self.update()
            else:
                # If dialog was cancelled, clear the drawing points
                self.drawing_points = []
                self.update()
