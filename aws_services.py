import cv2
import numpy as np
import boto3
from botocore.exceptions import ClientError
from datetime import datetime

class AWSRekognitionService:
    """AWS Rekognition service for person detection"""
    
    def __init__(self):
        self.rekognition_client = None
        self.aws_enabled = False
        self.api_calls_count = 0
        self.last_api_call_time = 0.0
        self.detection_interval = 4.0  # AWS detection every 4 seconds
        self.last_detection_time = 0.0
    
    def enable_aws_rekognition(self, aws_region='ap-south-1'):
        """Enable AWS Rekognition using default credentials"""
        try:
            # Use default credentials with specified region
            self.rekognition_client = boto3.client('rekognition', region_name=aws_region)
            
            # Test the connection with minimal API call
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            _, test_bytes = cv2.imencode('.jpg', test_image)
            
            self.rekognition_client.detect_labels(
                Image={'Bytes': test_bytes.tobytes()},
                MaxLabels=1,
                MinConfidence=90.0
            )
            
            self.aws_enabled = True
            self.api_calls_count = 0
            self.last_api_call_time = None
            print("AWS Rekognition enabled - API calls will be made every 2 seconds")
            return True, "AWS Rekognition enabled successfully"
            
        except ClientError as e:
            self.aws_enabled = False
            self.rekognition_client = None
            return False, f"AWS Rekognition error: {str(e)}"
        except Exception as e:
            self.aws_enabled = False
            self.rekognition_client = None
            return False, f"Error enabling AWS Rekognition: {str(e)}"
    
    def detect_people(self, frame, current_time):
        """Detect people using AWS Rekognition"""
        if not self.aws_enabled:
            return []
        
        # Use AWS for detection if enough time has passed
        if self.should_detect(current_time):
            try:
                # Convert frame to JPEG bytes with reduced quality for cost optimization
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_param)
                
                # Call AWS Rekognition with optimized parameters
                response = self.rekognition_client.detect_labels(
                    Image={'Bytes': jpeg_bytes.tobytes()},
                    MaxLabels=5,
                    MinConfidence=75.0
                )
                
                # Update API call tracking
                self.api_calls_count += 1
                self.last_api_call_time = current_time
                print(f"AWS API call #{self.api_calls_count} at time {current_time:.2f}s")
                
                # Filter for people and extract bounding boxes
                people = []
                for label in response['Labels']:
                    if label['Name'].lower() == 'person':
                        for instance in label.get('Instances', []):
                            if instance['Confidence'] >= 75.0:
                                bbox = instance['BoundingBox']
                                x = int(bbox['Left'] * frame.shape[1])
                                y = int(bbox['Top'] * frame.shape[0])
                                width = int(bbox['Width'] * frame.shape[1])
                                height = int(bbox['Height'] * frame.shape[0])
                                
                                margin = 15
                                x = max(0, x - margin)
                                y = max(0, y - margin)
                                width = min(frame.shape[1] - x, width + 2 * margin)
                                height = min(frame.shape[0] - y, height + 2 * margin)
                                
                                people.append({
                                    'bbox': (x, y, width, height),
                                    'confidence': instance['Confidence'],
                                    'detection_type': 'aws',
                                    'timestamp': current_time
                                })
                
                self.update_detection_time(current_time)
                return people
                
            except Exception as e:
                print(f"Error in AWS person detection: {str(e)}")
                return []
        
        return []
    
    def should_detect(self, current_time):
        """Check if enough time has passed since last detection"""
        time_since_last_detection = current_time - self.last_detection_time
        return time_since_last_detection >= self.detection_interval
    
    def update_detection_time(self, current_time):
        """Update the last detection time"""
        self.last_detection_time = current_time
    
    def get_api_stats(self):
        """Get API usage statistics"""
        return {
            'calls_count': self.api_calls_count,
            'last_call_time': self.last_api_call_time,
            'aws_enabled': self.aws_enabled
        }
    
    def get_detection_color(self, detection_type, current_time):
        """Get color for drawing based on detection type"""
        return (0, 0, 255)  # Red for all detections (AWS only now)