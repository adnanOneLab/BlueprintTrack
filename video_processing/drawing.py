from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QFont
import numpy as np
from PyQt6.QtCore import Qt, QPoint

class DrawingMixin:
    def paintEvent(self, event):
        """Draw the video frame and overlays"""
        if self.scaled_frame is None:
            return
        
        painter = QPainter(self)
        
        # Draw video frame
        height, width = self.scaled_frame.shape[:2]
        qimage = QImage(self.scaled_frame.data, width, height,
                       self.scaled_frame.strides[0], QImage.Format.Format_RGB888)
        painter.drawImage(self.frame_offset_x, self.frame_offset_y, qimage)
        
        # Only draw tracked people during export mode with additional validation
        if self.is_exporting and hasattr(self, 'aws_service') and self.aws_service.is_exporting and self.tracked_people:
            current_frame = getattr(self, 'current_frame_number', 0)
            valid_tracks_drawn = 0
            
            for person_id, person in self.tracked_people.items():
                # Validate track before drawing
                if not self._is_valid_track(person, current_frame):
                    continue
                
                x, y, w, h = person['bbox']
                current_store = person.get('current_store')
                
                # Validate bounding box coordinates
                if not self._is_valid_bbox(x, y, w, h):
                    continue
                
                # Scale coordinates to widget size
                scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                scaled_w = int(w * self.scale_factor)
                scaled_h = int(h * self.scale_factor)
                
                # Additional validation for scaled coordinates
                if scaled_w <= 0 or scaled_h <= 0:
                    continue
                
                # Calculate confidence-based color intensity
                is_moving = person.get('is_moving', False)
                if is_moving:
                    pen_color = QColor(255, 165, 0)  # Orange for moving
                else:
                    pen_color = QColor(0, 0, 255)    # Blue for idle
                painter.setPen(QPen(pen_color, 2))
                painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
                
                # Draw label with store info if in store
                if current_store is not None and current_store in self.stores:
                    store_name = self.stores[current_store].get('name', 'Unknown')
                    # Truncate store name if too long
                    store_display = store_name[:8] if len(store_name) > 8 else store_name
                    label = f"ID:{person_id}|{store_display}"
                else:
                    label = f"ID:{person_id}|OUT"

                status = "Moving" if person.get('is_moving', False) else "Idle"
                label += f"|{status}"
                
                painter.setFont(QFont("Arial", 10))
                painter.setPen(QColor(255, 255, 255))
                
                # Draw text background
                text_rect = painter.fontMetrics().boundingRect(label)
                text_rect.moveTop(scaled_y - text_rect.height() - 2)
                text_rect.moveLeft(scaled_x)
                text_rect.adjust(-2, -1, 2, 1)
                painter.fillRect(text_rect, QColor(0, 0, 0, 200))
                
                # Draw text
                painter.drawText(scaled_x, scaled_y - 4, label)
                
                valid_tracks_drawn += 1
            
            # Debug info
            if hasattr(self, 'debug_mode') and self.debug_mode and valid_tracks_drawn > 0:
                debug_text = f"Active: {valid_tracks_drawn}/{len(self.tracked_people)}"
                painter.setFont(QFont("Arial", 12))
                painter.setPen(QColor(255, 255, 0))
                painter.drawText(10, 30, debug_text)
        
        # Draw calibration points and lines if in calibration mode
        if hasattr(self, 'calibration_mode') and self.calibration_mode:
            self._draw_calibration_overlay(painter)
        
        # Draw store polygons in test mode
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'stores') and self.stores:
            self._draw_store_polygons(painter)
    
    def _is_valid_track(self, person, current_frame):
        """Validate if a track is valid for drawing"""
        # Check if track has required fields
        if 'bbox' not in person or 'last_seen' not in person:
            return False
        
        # Check if track is recent enough
        frames_since_seen = current_frame - person['last_seen']
        if frames_since_seen > 5:  # Only draw very recent tracks
            return False
        
        # Check confidence
        confidence = person.get('confidence', 0)
        if confidence < 50:  # Minimum confidence for drawing
            return False
        
        return True
    
    def _is_valid_bbox(self, x, y, w, h):
        """Validate bounding box coordinates"""
        # Check for negative or zero dimensions
        if w <= 0 or h <= 0:
            return False
        
        # Check for reasonable size (not too small or too large)
        if w < 10 or h < 20 or w > 1000 or h > 1000:
            return False
        
        # Check for reasonable position
        if x < -100 or y < -100 or x > 2000 or y > 2000:
            return False
        
        return True
    
    def _draw_calibration_overlay(self, painter):
        """Draw calibration points and lines"""
        if not hasattr(self, 'calibration_points'):
            return
        
        # Draw existing points
        for i, point in enumerate(self.calibration_points):
            x = int(point[0] * self.scale_factor + self.frame_offset_x)
            y = int(point[1] * self.scale_factor + self.frame_offset_y)
            
            # Draw point
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.drawEllipse(QPoint(x, y), 5, 5)
            
            # Draw label
            painter.setFont(QFont("Arial", 17, QFont.Weight.Bold))
            painter.setPen(QColor(255, 255, 255))
            
            # Draw text background
            if hasattr(self, 'calibration_point_labels') and i < len(self.calibration_point_labels):
                text = f"{i+1}. {self.calibration_point_labels[i]}"
                text_rect = painter.fontMetrics().boundingRect(text)
                text_rect.moveCenter(QPoint(x, y - 20))
                text_rect.adjust(-5, -2, 5, 2)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(x, y - 20, text)
        
        # Draw lines between points
        if len(self.calibration_points) > 1:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            for i in range(len(self.calibration_points) - 1):
                x1 = int(self.calibration_points[i][0] * self.scale_factor + self.frame_offset_x)
                y1 = int(self.calibration_points[i][1] * self.scale_factor + self.frame_offset_y)
                x2 = int(self.calibration_points[i+1][0] * self.scale_factor + self.frame_offset_x)
                y2 = int(self.calibration_points[i+1][1] * self.scale_factor + self.frame_offset_y)
                painter.drawLine(x1, y1, x2, y2)
            
            # Draw line from last point to first point if we have 3 points
            if len(self.calibration_points) == 3:
                x1 = int(self.calibration_points[-1][0] * self.scale_factor + self.frame_offset_x)
                y1 = int(self.calibration_points[-1][1] * self.scale_factor + self.frame_offset_y)
                x2 = int(self.calibration_points[0][0] * self.scale_factor + self.frame_offset_x)
                y2 = int(self.calibration_points[0][1] * self.scale_factor + self.frame_offset_y)
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw next point indicator
        if hasattr(self, 'calibration_point_labels') and len(self.calibration_points) < len(self.calibration_point_labels):
            next_point = self.calibration_point_labels[len(self.calibration_points)]
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.setPen(QColor(255, 255, 255))
            
            # Draw text background
            text = f"Click to place {next_point} point"
            text_rect = painter.fontMetrics().boundingRect(text)
            text_rect.moveCenter(QPoint(self.width() // 2, 30))
            text_rect.adjust(-10, -5, 10, 5)
            painter.fillRect(text_rect, QColor(0, 0, 0, 180))
            # Draw text
            painter.drawText(self.width() // 2, 30, text)
    
    def _draw_store_polygons(self, painter):
        """Draw store polygons in test mode"""
        # Debug information
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Drawing {len(self.stores)} stores in test mode")
        
        # Pre-calculate all store polygon data for performance
        store_polygon_data = []
        for store_id, store in self.stores.items():
            if "video_polygon" in store and len(store["video_polygon"]) > 2:
                try:
                    # Scale the transformed points to widget coordinates
                    video_polygon = []
                    for x, y in store["video_polygon"]:
                        scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                        scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                        video_polygon.append((scaled_x, scaled_y))
                    
                    if len(video_polygon) > 2:
                        # Pre-calculate centroid
                        centroid_x = int(np.mean([p[0] for p in video_polygon]))
                        centroid_y = int(np.mean([p[1] for p in video_polygon]))
                        
                        store_polygon_data.append({
                            'polygon': video_polygon,
                            'centroid_x': centroid_x,
                            'centroid_y': centroid_y,
                            'name': store.get("name", "")
                        })
                
                except Exception as e:
                    # Handle any errors that occur during polygon drawing
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"Error pre-calculating polygon data for store {store_id}: {str(e)}")
                    # Continue with next store instead of crashing
                    continue
        
        # Draw all store polygons using pre-calculated data
        for polygon_data in store_polygon_data:
            try:
                video_polygon = polygon_data['polygon']
                
                # Draw filled semi-transparent polygon
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 255, 0, 50))  # Semi-transparent green
                painter.drawPolygon([QPoint(x, y) for x, y in video_polygon])
                
                # Draw polygon outline
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for i in range(len(video_polygon) - 1):
                    painter.drawLine(video_polygon[i][0], video_polygon[i][1],
                                   video_polygon[i+1][0], video_polygon[i+1][1])
                painter.drawLine(video_polygon[-1][0], video_polygon[-1][1],
                               video_polygon[0][0], video_polygon[0][1])
                
                # Draw store name
                if polygon_data['name']:
                    painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
                    
                    # Draw text background
                    text_rect = painter.fontMetrics().boundingRect(polygon_data['name'])
                    text_rect.moveCenter(QPoint(polygon_data['centroid_x'], polygon_data['centroid_y']))
                    text_rect.adjust(-5, -2, 5, 2)
                    painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                    
                    # Draw text
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(polygon_data['centroid_x'], polygon_data['centroid_y'], polygon_data['name'])
            
            except Exception as e:
                # Handle any errors that occur during polygon drawing
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"Error drawing store polygon: {str(e)}")
                # Continue with next store instead of crashing
                continue