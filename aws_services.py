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
        self.last_api_call_time = None
        self.detection_interval = 2.0  # Process every 2 seconds
        self.last_detection_time = 0
    
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
    
    def detect_people(self, frame):
        """Detect people in the frame using AWS Rekognition"""
        if not self.aws_enabled or self.rekognition_client is None:
            return []
        
        try:
            # Convert frame to JPEG bytes with reduced quality for cost optimization
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Reduce quality to 85%
            _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_param)
            
            # Call AWS Rekognition with optimized parameters
            response = self.rekognition_client.detect_labels(
                Image={'Bytes': jpeg_bytes.tobytes()},
                MaxLabels=5,  # Reduced from 10 to 5 since we only care about people
                MinConfidence=75.0  # Increased confidence threshold
            )
            
            # Update API call tracking
            self.api_calls_count += 1
            self.last_api_call_time = datetime.now()
            
            # Log API usage periodically
            if self.api_calls_count % 10 == 0:  # Log every 10 calls
                print(f"AWS API calls: {self.api_calls_count} (Last call: {self.last_api_call_time})")
            
            # Filter for people and extract bounding boxes
            people = []
            for label in response['Labels']:
                if label['Name'].lower() == 'person':
                    for instance in label.get('Instances', []):
                        if instance['Confidence'] >= 75.0:  # Increased confidence threshold
                            bbox = instance['BoundingBox']
                            # Convert normalized coordinates to pixel coordinates
                            x = int(bbox['Left'] * frame.shape[1])
                            y = int(bbox['Top'] * frame.shape[0])
                            width = int(bbox['Width'] * frame.shape[1])
                            height = int(bbox['Height'] * frame.shape[0])
                            confidence = instance['Confidence']
                            people.append({
                                'bbox': (x, y, width, height),
                                'confidence': confidence
                            })
            
            return people
            
        except Exception as e:
            print(f"Error in person detection: {str(e)}")
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

# Legacy functions for backward compatibility
def detect_people(self, frame):
    """Legacy function - use AWSRekognitionService.detect_people instead"""
    if hasattr(self, 'aws_service'):
        return self.aws_service.detect_people(frame)
    return []

def enable_aws_rekognition(self, aws_region='ap-south-1'):
    """Legacy function - use AWSRekognitionService.enable_aws_rekognition instead"""
    if not hasattr(self, 'aws_service'):
        self.aws_service = AWSRekognitionService()
    success, message = self.aws_service.enable_aws_rekognition(aws_region)
    if hasattr(self, 'status_message'):
        self.status_message.emit(message)
    return success