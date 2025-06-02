import cv2
import numpy as np
from PyQt6.QtWidgets import QMessageBox

class CalibrationMixin:
    def _init_calibration(self):
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.test_mode = False
        self.stores = {}
        self.store_perspective_matrices = {}

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
