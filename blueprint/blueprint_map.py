import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class BlueprintMapMixin:
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
