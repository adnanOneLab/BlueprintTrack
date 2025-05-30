"""
Blueprint Processing for CCTV-Store Mapping (GUI Ready)

This code converts a mall blueprint image with marked cameras into a spatial data format
that can be used by the CCTV-Store mapping system.

Features:
- Load and process blueprint images
- Convert between pixel and meter coordinates
- Add cameras and stores manually or automatically
- Export/import spatial data to/from JSON
- Visualize the mapped layout
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
import math
import cv2
import numpy as np
import json
from PyQt6.QtWidgets import (QWidget, QMessageBox, QFormLayout, 
                             QSizePolicy, QDialog, QLineEdit, QDialogButtonBox)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QFont


class BlueprintProcessor:
    """Main class for processing blueprint images and managing spatial data"""
    
    def __init__(self):
        # Image and display properties
        self.image = None
        self.display_image = None
        
        # Coordinate system properties
        self.scale_factor = 1.0  # meters per pixel
        self.origin = (0, 0)     # (x, y) coordinate of origin in pixels
        self.floor = 1           # Current floor number
        
        # Spatial data storage
        self.cameras = {}
        self.stores = {}

    # ==========================================
    # BLUEPRINT LOADING AND SETUP
    # ==========================================
    
    def load_blueprint(self, image_path):
        """Load the blueprint image from file"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for display
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        print(f"Blueprint loaded: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        return True
    
    def set_scale(self, pixels, meters):
        """Set the scale factor (meters per pixel)"""
        self.scale_factor = meters / pixels
        print(f"Scale set: {self.scale_factor:.4f} meters per pixel")
        return True
    
    def set_origin(self, x, y):
        """Set the origin point (in pixels)"""
        self.origin = (x, y)
        print(f"Origin set: pixel coordinates ({x}, {y})")
        return True
    
    def set_floor(self, floor_number):
        """Set the floor number for the current blueprint"""
        self.floor = floor_number
        print(f"Floor set: {floor_number}")
        return True

    # ==========================================
    # COORDINATE CONVERSION
    # ==========================================
    
    def pixels_to_meters(self, px, py):
        """Convert pixel coordinates to meter coordinates"""
        rel_x = px - self.origin[0]
        rel_y = self.origin[1] - py  # Invert y-axis
        
        meter_x = rel_x * self.scale_factor
        meter_y = rel_y * self.scale_factor
        
        return meter_x, meter_y
    
    def meters_to_pixels(self, mx, my):
        """Convert meter coordinates to pixel coordinates"""
        rel_x = mx / self.scale_factor
        rel_y = my / self.scale_factor
        
        px = int(self.origin[0] + rel_x)
        py = int(self.origin[1] - rel_y)  # Invert y-axis
        
        return px, py

    # ==========================================
    # CAMERA MANAGEMENT
    # ==========================================
    
    def add_camera_manual(self, camera_id, pixel_x, pixel_y, orientation, 
                         fov_angle=70, fov_range=20):
        """Add a camera manually by specifying position and orientation"""
        meter_x, meter_y = self.pixels_to_meters(pixel_x, pixel_y)
        
        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "location": {
                "x": meter_x,
                "y": meter_y,
                "floor": self.floor,
                "orientation": orientation
            },
            "field_of_view": {
                "angle": fov_angle,
                "range": fov_range
            },
            "resolution": {
                "width": 1920,
                "height": 1080
            },
            "status": "active"
        }
        
        print(f"Camera {camera_id} added: position ({meter_x:.2f}, {meter_y:.2f})m, orientation {orientation}°")
        return True
    
    def detect_cameras_by_color(self, lower_color, upper_color, min_area=100):
        """Automatically detect cameras based on color markers"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        camera_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Get center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    camera_id = f"cam{camera_count+1:03d}"
                    self.add_camera_manual(camera_id, cx, cy, 0)  # Default orientation
                    camera_count += 1
        
        print(f"Auto-detected {camera_count} cameras from color markers")
        return camera_count

    # ==========================================
    # STORE MANAGEMENT
    # ==========================================
    
    def add_store_manual(self, store_id, name, category, pixel_polygon, metadata=None):
        """Add a store manually by specifying its boundary polygon"""
        # Convert pixel polygon to meter coordinates
        meter_polygon = []
        for px, py in pixel_polygon:
            mx, my = self.pixels_to_meters(px, py)
            meter_polygon.append({"x": mx, "y": my})
        
        self.stores[store_id] = {
            "store_id": store_id,
            "name": name,
            "category": category,
            "location": {
                "floor": self.floor,
                "polygon": meter_polygon
            },
            "metadata": metadata or {}
        }
        
        print(f"Store {store_id} '{name}' added: {len(pixel_polygon)} boundary points")
        return True

    # ==========================================
    # VISUALIZATION
    # ==========================================
    
    def draw_cameras(self, img):
        """Draw cameras and their field of view on the image"""
        for camera_id, camera in self.cameras.items():
            px, py = self.meters_to_pixels(
                camera["location"]["x"], 
                camera["location"]["y"]
            )
            
            # Draw camera marker
            cv2.circle(img, (px, py), 5, (255, 0, 0), -1)
            cv2.putText(img, camera_id, (px + 10, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw field of view cone
            orientation = camera["location"]["orientation"]
            fov_angle = camera["field_of_view"]["angle"]
            fov_range_meters = camera["field_of_view"]["range"]
            fov_range_pixels = int(fov_range_meters / self.scale_factor)
            
            # Calculate cone edges
            half_angle = fov_angle / 2
            angle1 = math.radians(orientation - half_angle)
            angle2 = math.radians(orientation + half_angle)
            
            edge1_x = px + int(fov_range_pixels * math.cos(angle1))
            edge1_y = py - int(fov_range_pixels * math.sin(angle1))
            edge2_x = px + int(fov_range_pixels * math.cos(angle2))
            edge2_y = py - int(fov_range_pixels * math.sin(angle2))
            
            # Draw cone lines and arc
            cv2.line(img, (px, py), (edge1_x, edge1_y), (255, 0, 0), 1)
            cv2.line(img, (px, py), (edge2_x, edge2_y), (255, 0, 0), 1)
            cv2.ellipse(img, (px, py), (fov_range_pixels, fov_range_pixels), 
                       0, -orientation + half_angle, -orientation - half_angle, 
                       (255, 0, 0), 1)
    
    def draw_stores(self, img):
        """Draw store boundaries on the image"""
        for store_id, store in self.stores.items():
            # Convert meter coordinates to pixels
            pixel_polygon = []
            for point in store["location"]["polygon"]:
                px, py = self.meters_to_pixels(point["x"], point["y"])
                pixel_polygon.append((px, py))
            
            # Draw store polygon
            pixel_polygon = np.array(pixel_polygon, np.int32)
            pixel_polygon = pixel_polygon.reshape((-1, 1, 2))
            cv2.polylines(img, [pixel_polygon], True, (0, 255, 0), 2)
            
            # Draw store name at centroid
            centroid_x = np.mean([p[0] for p in pixel_polygon.reshape(-1, 2)])
            centroid_y = np.mean([p[1] for p in pixel_polygon.reshape(-1, 2)])
            cv2.putText(img, store["name"], (int(centroid_x), int(centroid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def visualize_map(self):
        """Create and display the complete map visualization"""
        display_img = self.display_image.copy()
        
        self.draw_cameras(display_img)
        self.draw_stores(display_img)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(display_img)
        plt.title('Mall Blueprint with Cameras and Stores')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return display_img

    # ==========================================
    # DATA EXPORT/IMPORT
    # ==========================================
    
    def export_to_json(self, output_file):
        """Export the spatial data to a JSON file"""
        data = {
            "mall_info": {
                "name": "Mall Blueprint",
                "floors": [self.floor],
                "scale_factor": self.scale_factor,
                "origin": {"x": self.origin[0], "y": self.origin[1]}
            },
            "cameras": list(self.cameras.values()),
            "stores": list(self.stores.values()),
            "reference_points": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {output_file}")
        return True
    
    def import_from_json(self, input_file):
        """Import spatial data from a JSON file"""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Load mall info
        if "mall_info" in data:
            mall_info = data["mall_info"]
            if "scale_factor" in mall_info:
                self.scale_factor = mall_info["scale_factor"]
            if "origin" in mall_info:
                self.origin = (mall_info["origin"]["x"], mall_info["origin"]["y"])
            if "floors" in mall_info and len(mall_info["floors"]) > 0:
                self.floor = mall_info["floors"][0]
        
        # Load cameras and stores
        if "cameras" in data:
            self.cameras = {camera["camera_id"]: camera for camera in data["cameras"]}
        if "stores" in data:
            self.stores = {store["store_id"]: store for store in data["stores"]}
        
        print(f"Data imported from {input_file}")
        print(f"Loaded: {len(self.cameras)} cameras, {len(self.stores)} stores")
        return True

    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_camera_count(self):
        """Get the current number of cameras"""
        return len(self.cameras)
    
    def get_store_count(self):
        """Get the current number of stores"""
        return len(self.stores)
    
    def clear_cameras(self):
        """Remove all cameras"""
        count = len(self.cameras)
        self.cameras.clear()
        print(f"Cleared {count} cameras")
        return True
    
    def clear_stores(self):
        """Remove all stores"""
        count = len(self.stores)
        self.stores.clear()
        print(f"Cleared {count} stores")
        return True
    
    def remove_camera(self, camera_id):
        """Remove a specific camera"""
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            print(f"Camera {camera_id} removed")
            return True
        return False
    
    def remove_store(self, store_id):
        """Remove a specific store"""
        if store_id in self.stores:
            del self.stores[store_id]
            print(f"Store {store_id} removed")
            return True
        return False
    

class BlueprintView(QWidget):
    """Custom widget for displaying and interacting with the blueprint"""
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