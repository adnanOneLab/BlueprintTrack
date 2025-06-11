#!/usr/bin/env python3
"""
Pelco VideoXpert Official API Snapshot Capture Tool - Fixed Version
"""

import os
import sys
import time
import base64
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import warnings

# Suppress only the InsecureRequestWarning from urllib3
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Configuration - Adjust these values
CONFIG = {
    "base_url": "https://190.116.49.5",
    "username": "wise",
    "password": "W1s3$2025",
    "camera_uuid": "98e87943-e4a1-88cc-743b-7737c1514f04",
    "api_version": "5.11",
    "output_dir": "test_scripts/snapshot_download_test/pelco_snapshots",
    "min_image_size": 150000,  # Minimum expected JPEG size in bytes
    "capture_interval": 10,    # Seconds between captures
    "historical_offset": 15,   # Minutes in past to capture from archive
    "max_retries": 3,
    "verify_ssl": False,       # Set to True with valid SSL cert
    "timeout": 15              # Request timeout in seconds
}

class VideoXpertSnapshot:
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.setup_headers()
        self.create_output_dir()
        
    def setup_headers(self):
        """Set up authentication headers"""
        encoded_user = base64.b64encode(self.config["username"].encode()).decode()
        encoded_pass = base64.b64encode(self.config["password"].encode()).decode()
        
        self.session.headers.update({
            "X-Serenity-User": encoded_user,
            "X-Serenity-Password": encoded_pass,
            "Accept": f"image/jpeg, application/vnd.pelco.resource+json; version={self.config['api_version']}",
            "User-Agent": "PelcoSnapshotClient/1.0"
        })

    def create_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        print(f"[+] Output directory: {os.path.abspath(self.config['output_dir'])}")

    def validate_image(self, image_data: bytes) -> bool:
        """Validate JPEG meets quality standards"""
        # Check minimum size requirement
        if len(image_data) < self.config["min_image_size"]:
            return False
        
        # Basic JPEG signature validation
        return image_data.startswith(b'\xff\xd8') and image_data.endswith(b'\xff\xd9')

    def get_snapshot(self, historical: bool = True) -> Optional[bytes]:
        """
        Get snapshot using official API
        :param historical: If True, gets from archive with offset
        :return: JPEG image data or None if failed
        """
        url = f"{self.config['base_url']}/system/{self.config['api_version']}/snapshots/uuid:{self.config['camera_uuid']}:video"
        
        params = {
            "quality": "high",
            "resolution": "full"
        }
        
        if historical:
            # Use timezone-aware UTC timestamp
            snapshot_time = datetime.now(timezone.utc) - timedelta(minutes=self.config["historical_offset"])
            params["time"] = snapshot_time.isoformat(timespec='seconds')

        try:
            response = self.session.get(
                url,
                params=params,
                verify=self.config["verify_ssl"],
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 200:
                if self.validate_image(response.content):
                    return response.content
                else:
                    print("[-] Image validation failed (size or format)")
            else:
                print(f"[-] API returned status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"[-] Request failed: {str(e)}")
            
        return None

    def capture_with_retry(self) -> Tuple[bool, str]:
        """Attempt to capture snapshot with retries"""
        for attempt in range(1, self.config["max_retries"] + 1):
            print(f"Attempt {attempt}/{self.config['max_retries']}...", end=" ", flush=True)
            image_data = self.get_snapshot()
            
            if image_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pelco_{self.config['camera_uuid'][:8]}_{timestamp}.jpg"
                filepath = os.path.join(self.config["output_dir"], filename)
                
                try:
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    
                    file_size = len(image_data) / 1024  # Convert to KB
                    return True, f"Saved {filename} ({file_size:.1f} KB)"
                except IOError as e:
                    return False, f"File save error: {str(e)}"
            
            time.sleep(2)  # Brief delay between attempts
            
        return False, "Failed to capture snapshot after retries"

    def run_continuous_capture(self):
        """Run continuous capture loop"""
        print("\n[ Pelco VideoXpert Snapshot Capture ]")
        print(f"Camera UUID: {self.config['camera_uuid']}")
        print(f"Capture interval: {self.config['capture_interval']} seconds")
        print(f"Historical offset: {self.config['historical_offset']} minutes\n")
        
        count = 1
        success_count = 0
        
        try:
            while True:
                start_time = time.time()
                print(f"\nCapture #{count}: ", end="", flush=True)
                
                success, message = self.capture_with_retry()
                if success:
                    success_count += 1
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
                
                count += 1
                
                # Calculate remaining wait time
                elapsed = time.time() - start_time
                remaining_wait = max(0, self.config["capture_interval"] - elapsed)
                if remaining_wait > 0:
                    time.sleep(remaining_wait)
                
        except KeyboardInterrupt:
            print("\n[!] Capture stopped by user")
        finally:
            print(f"\nSummary: {success_count} successful captures out of {count-1} attempts")

if __name__ == "__main__":
    try:
        vx = VideoXpertSnapshot(CONFIG)
        vx.run_continuous_capture()
    except Exception as e:
        print(f"[!] Fatal error: {e}")
        sys.exit(1)