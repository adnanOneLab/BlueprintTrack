import requests
import base64
import time
import os
from datetime import datetime

# --- USER CONFIGURABLE SECTION ---
username = "wise"
password = "W1s3$2025"

# Set when to start capturing snapshots (in 12-hour format)
START_DATE = "06/05/2025"       # Format: YYYY-MM-DD
START_TIME = "02:15:00 PM"      # Format: HH:MM:SS AM/PM
# ----------------------------------

# Encode credentials
encoded_user = base64.b64encode(username.encode()).decode()
encoded_password = base64.b64encode(password.encode()).decode()

BASE_URL = "https://190.116.49.5"
SNAPSHOT_UUID = "98e87943-e4a1-88cc-743b-7737c1514f04"
BLOB_URL = "blob:https://190.116.49.5/b0556aba-5d21-4404-83cf-521b56cf91f4"

# Create snapshots directory
SNAPSHOTS_DIR = "test_scripts/snapshot_download_test/snapshots"
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)
    print(f"[+] Created directory: {SNAPSHOTS_DIR}")

HEADERS_LOGIN = {
    "X-Serenity-User": encoded_user,
    "X-Serenity-Password": encoded_password,
    "Accept": "application/vnd.pelco.resource+json; version=5.11",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
}

HEADERS_SNAPSHOT = {
    "X-Serenity-User": encoded_user,
    "X-Serenity-Password": encoded_password,
    "Accept": "image/jpeg",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
}

requests.packages.urllib3.disable_warnings()

def login_and_get_session():
    session = requests.Session()
    session.headers.update(HEADERS_LOGIN)
    url = f"{BASE_URL}/system"
    response = session.get(url, verify=False)
    if response.status_code == 200:
        print("[+] Login successful or session valid.")
        return session
    else:
        print(f"[-] Login failed, status code: {response.status_code}")
        print(response.text)
        return None

def download_from_blob(session):
    try:
        response = session.get(BLOB_URL, headers=HEADERS_SNAPSHOT, verify=False)
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SNAPSHOTS_DIR, f"blob_snapshot_{timestamp}.jpg")
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"[+] Blob snapshot saved as {filename}")
            return True
        else:
            print(f"[-] Failed to download from blob URL, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[-] Error downloading from blob URL: {e}")
        return False

def capture_screenshot(session, count=None):
    snapshot_url = f"{BASE_URL}/system/5.11/snapshots/uuid:{SNAPSHOT_UUID}:video"
    response = session.get(snapshot_url, headers=HEADERS_SNAPSHOT, verify=False)

    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SNAPSHOTS_DIR, f"snapshot_{count:04d}_{timestamp}.jpg" if count else f"snapshot_{timestamp}.jpg")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"[+] Snapshot saved as {filename}")
        return True
    else:
        print(f"[-] Failed to capture snapshot, status code: {response.status_code}")
        print(response.text)
        return False

def wait_until_start_time():
    try:
        start_datetime_str = f"{START_DATE} {START_TIME}"
        start_datetime = datetime.strptime(start_datetime_str, "%m/%d/%Y %I:%M:%S %p")
        now = datetime.now()

        if now < start_datetime:
            wait_time = (start_datetime - now).total_seconds()
            print(f"[⏳] Waiting for start time: {start_datetime} (in {int(wait_time)} seconds)")
            time.sleep(wait_time)
        else:
            print(f"[⚠️] Start time {start_datetime} already passed. Starting immediately.")
    except Exception as e:
        print(f"[!] Failed to parse start time: {e}")
        exit(1)

def main():
    session = login_and_get_session()
    if not session:
        return

    wait_until_start_time()

    print("[+] Attempting to download from blob URL first...")
    download_from_blob(session)

    print("[+] Starting continuous snapshot capture...")
    count = 1
    success_count = 0

    try:
        while True:
            if capture_screenshot(session, count):
                success_count += 1
            count += 1

            if count % 10 == 0:
                print(f"[INFO] Captured {success_count} out of {count - 1} attempts")

            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\n[!] Stopped by user. Total successful captures: {success_count}")

if __name__ == "__main__":
    main()
