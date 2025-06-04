import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import os

class AWSRekognitionService:
    """AWS Rekognition service for face detection only"""
    
    def __init__(self):
        self.rekognition_client = None
        self.aws_enabled = False
        self.api_calls_count = 0
        self.last_api_call_time = 0.0
        self.face_detection_interval = 2.0  # Face detection every 2 seconds
        self.last_face_detection_time = 0.0
        self.body_images_dir = os.path.join(os.getcwd(), "exports/body_images/")
        os.makedirs(self.body_images_dir, exist_ok=True)
        self.is_exporting = False  # Track if we're in export mode
        self.last_face_detections = []  # Store last face detections
    
    def enable_aws_rekognition(self, aws_region='ap-south-1'):
        """Enable AWS Rekognition using default credentials"""
        try:
            # Use default credentials with specified region
            self.rekognition_client = boto3.client('rekognition', region_name=aws_region)
            
            # Test the connection with minimal API call
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            _, test_bytes = cv2.imencode('.jpg', test_image)
            
            self.rekognition_client.detect_faces(
                Image={'Bytes': test_bytes.tobytes()},
                Attributes=['ALL']
            )
            
            self.aws_enabled = True
            self.api_calls_count = 0
            self.last_api_call_time = None
            print("AWS Rekognition enabled - Face detection API calls will be made every 2 seconds")
            return True, "AWS Rekognition enabled successfully"
            
        except ClientError as e:
            self.aws_enabled = False
            self.rekognition_client = None
            return False, f"AWS Rekognition error: {str(e)}"
        except Exception as e:
            self.aws_enabled = False
            self.rekognition_client = None
            return False, f"Error enabling AWS Rekognition: {str(e)}"
    
    def set_export_mode(self, enabled):
        """Enable/disable export mode to control API calls"""
        self.is_exporting = enabled
        if enabled:
            print("AWS Rekognition enabled for face detection in export mode")
        else:
            print("AWS Rekognition disabled for preview mode")
    
    def detect_faces(self, frame, current_time):
        """Detect faces using AWS Rekognition"""
        if not self.aws_enabled or not self.is_exporting:
            return []
        
        # Use AWS for face detection if enough time has passed
        if current_time - self.last_face_detection_time >= self.face_detection_interval:
            try:
                print(f"\nAttempting face detection at time {current_time:.2f}s")
                # Convert frame to JPEG bytes with reduced quality for cost optimization
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Increased quality for better face detection
                _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_param)
                
                # Call AWS Rekognition for face detection with more lenient settings
                response = self.rekognition_client.detect_faces(
                    Image={'Bytes': jpeg_bytes.tobytes()},
                    Attributes=['ALL']
                )
                
                # Update API call tracking
                self.api_calls_count += 1
                self.last_api_call_time = current_time
                self.last_face_detection_time = current_time
                
                # Extract face detections and capture body images
                faces = []
                face_count = len(response.get('FaceDetails', []))
                print(f"Found {face_count} faces in frame")
                
                for i, face in enumerate(response.get('FaceDetails', [])):
                    if face['Confidence'] >= 60.0:  # Lowered confidence threshold from 75%
                        bbox = face['BoundingBox']
                        x = int(bbox['Left'] * frame.shape[1])
                        y = int(bbox['Top'] * frame.shape[0])
                        width = int(bbox['Width'] * frame.shape[1])
                        height = int(bbox['Height'] * frame.shape[0])
                        
                        print(f"Face {i+1}: Confidence={face['Confidence']:.1f}%, Position=({x}, {y}, {width}, {height})")
                        
                        # Calculate expanded body region (larger area around face)
                        # Use face height as reference for body proportions
                        body_width = int(width * 3.0)  # Wider than face
                        body_height = int(height * 7.0)  # Taller than face
                        
                        # Center the body region horizontally on the face
                        body_x = max(0, x - (body_width - width) // 2)
                        body_x = min(frame.shape[1] - body_width, body_x)  # Don't exceed frame width
                        
                        # Position body region to include face in upper third
                        body_y = max(0, y - height)  # Start above face
                        body_y = min(frame.shape[0] - body_height, body_y)  # Don't exceed frame height
                        
                        # Ensure we don't exceed frame boundaries
                        body_width = min(body_width, frame.shape[1] - body_x)
                        body_height = min(body_height, frame.shape[0] - body_y)
                        
                        # Capture full-color body image
                        body_image = frame[body_y:body_y + body_height, body_x:body_x + body_width].copy()
                        
                        if body_image.size > 0:  # Check if image is valid
                            # Resize to a larger size while maintaining aspect ratio
                            target_height = 800  # Set target height
                            aspect_ratio = body_width / body_height
                            target_width = int(target_height * aspect_ratio)
                            
                            # Resize image with high-quality interpolation
                            body_image = cv2.resize(body_image, (target_width, target_height), 
                                                  interpolation=cv2.INTER_LANCZOS4)
                            
                            # Save full-color image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            image_path = os.path.join(self.body_images_dir, f"body_{timestamp}.jpg")
                            cv2.imwrite(image_path, cv2.cvtColor(body_image, cv2.COLOR_RGB2BGR))
                            print(f"  Saved body image to: {image_path} ({target_width}x{target_height})")
                        else:
                            print(f"  Warning: Could not capture body image for face {i+1} (invalid dimensions)")
                        
                        faces.append({
                            'bbox': (x, y, width, height),
                            'body_bbox': (body_x, body_y, body_width, body_height),
                            'confidence': face['Confidence'],
                            'detection_type': 'face',
                            'timestamp': current_time,
                            'image_path': image_path if 'image_path' in locals() else None
                        })
                    else:
                        print(f"Face {i+1}: Confidence too low ({face['Confidence']:.1f}% < 60%)")
                
                print(f"Successfully processed {len(faces)} faces above confidence threshold")
                self.last_face_detections = faces  # Store for drawing
                return faces
                
            except Exception as e:
                print(f"Error in AWS face detection: {str(e)}")
                return []
        else:
            time_since_last = current_time - self.last_face_detection_time
        
        return []
    
    def get_api_stats(self):
        """Get API usage statistics"""
        return {
            'calls_count': self.api_calls_count,
            'last_call_time': self.last_api_call_time,
            'aws_enabled': self.aws_enabled
        }
    
    def get_detection_color(self, detection_type, current_time):
        """Get color for drawing based on detection type"""
        if detection_type == 'face':
            return (0, 0, 255)  # Red for face detection
        return (255, 0, 0)  # Blue for other detections