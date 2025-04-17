import requests
import os
import zipfile
from datetime import datetime, timedelta

# Base URL and output directory
BASE_URL = "https://dp01012847.lolipop.jp/k_data"
OUTPUT_DIR = "./data"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date range for 2020 to 2024
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)
# start_date = datetime(2019, 1, 1)
# end_date = datetime(2019, 12, 31)

# Iterate through each date in the range
current_date = start_date
while current_date <= end_date:
    # Format the date parts for the URL and directory structure
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    date_str_short = current_date.strftime("%y%m%d")  # e.g., "230305"
    date_str_long = current_date.strftime("%Y%m%d")  # e.g., "20230305"
    url = f"{BASE_URL}/{year}/NS_{year[2:]}{month}/NSL_{date_str_short}.zip"
    
    # Create year/month subdirectory
    sub_dir = os.path.join(OUTPUT_DIR, year, month)
    os.makedirs(sub_dir, exist_ok=True)
    
    zip_path = os.path.join(sub_dir, f"{date_str_long}.zip")
    
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
            zip_ref.extractall(sub_dir)
        
        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"Saved and extracted: {zip_path}")
    except Exception as e:
        print(f"Failed to process {url}: {e}")
    
    # Move to the next day
    current_date += timedelta(days=1)
