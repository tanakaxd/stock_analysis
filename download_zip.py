import requests
import os
import zipfile
from datetime import datetime, timedelta

# Base URL and output directory
BASE_URL = "https://dp01012847.lolipop.jp/k_data/2024/NS_2404/NSL_" #日経先物のデータ
OUTPUT_DIR = "./data"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date range for April 2024
start_date = datetime(2024, 4, 1)
end_date = datetime(2024, 4, 30)

# Iterate through each date in the range
current_date = start_date
while current_date <= end_date:
    # Format the date part of the URL
    date_str = current_date.strftime("%y%m%d")
    url = f"{BASE_URL}{date_str}.zip"
    zip_path = os.path.join(OUTPUT_DIR, f"{date_str}.zip")
    
    try:
        # Download the zip file
        print(f"Downloading: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the zip file
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
        
        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"Saved and extracted: {zip_path}")
    except Exception as e:
        print(f"Failed to process {url}: {e}")
    
    # Move to the next day
    current_date += timedelta(days=1)
