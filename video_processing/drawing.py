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
        
        # Only draw tracked people during export mode
        if self.is_exporting and self.aws_service.is_exporting and self.tracked_people:
            for person_id, person in self.tracked_people.items():
                x, y, w, h = person['bbox']
                current_store = person['current_store']
                
                # Scale coordinates to widget size
                scaled_x = int(x * self.scale_factor + self.frame_offset_x)
                scaled_y = int(y * self.scale_factor + self.frame_offset_y)
                scaled_w = int(w * self.scale_factor)
                scaled_h = int(h * self.scale_factor)
                
                # Draw bounding box (red for all detections)
                painter.setPen(QPen(QColor(0, 0, 255), 2))
                painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
                
                # Draw label with store info if in store
                if current_store is not None:
                    store_name = self.stores[current_store]['name']
                    label = f"{person_id}|{store_name[:5]}"
                else:
                    label = f"{person_id}|OUT"
                
                painter.setFont(QFont("Arial", 12))
                painter.setPen(QColor(255, 255, 255))
                # Draw text background
                text_rect = painter.fontMetrics().boundingRect(label)
                text_rect.moveTop(scaled_y - text_rect.height())
                text_rect.moveLeft(scaled_x)
                text_rect.adjust(-1, -1, 1, 1)
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                # Draw text
                painter.drawText(scaled_x, scaled_y - 2, label)
        
        # Draw calibration points and lines if in calibration mode
        if self.calibration_mode:
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
            if len(self.calibration_points) < 4:
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
        
        # Draw store polygons in test mode
        if self.test_mode and self.stores:
            # Debug information
            print(f"Drawing {len(self.stores)} stores in test mode")
            
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
                            if "name" in store:
                                centroid_x = int(np.mean([p[0] for p in video_polygon]))
                                centroid_y = int(np.mean([p[1] for p in video_polygon]))
                                painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
                                # Draw text background
                                text_rect = painter.fontMetrics().boundingRect(store["name"])
                                text_rect.moveCenter(QPoint(centroid_x, centroid_y))
                                text_rect.adjust(-5, -2, 5, 2)
                                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                                # Draw text
                                painter.setPen(QColor(255, 255, 255))
                                painter.drawText(centroid_x, centroid_y, store["name"])
                    except Exception as e:
                        print(f"Error drawing store {store_id}: {str(e)}")
    