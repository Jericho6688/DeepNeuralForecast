import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import yaml
import os

with open("preprocess.yaml", "r") as f:
    config = yaml.safe_load(f)
crypt_ticker = config["data_download"]["crypt_ticker"]

start_date = datetime(2021, 9, 17)
end_date = datetime.now()
api_token = "demo"
interval = timedelta(days=120)
all_data = []

current_start = start_date
while current_start < end_date:
    current_end = min(current_start + interval, end_date)
    print(f"Downloading data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")

    start_timestamp = int(current_start.timestamp())
    end_timestamp = int(current_end.timestamp())
    url = f"https://eodhd.com/api/intraday/{crypt_ticker}-usd.cc?&from={start_timestamp}&to={end_timestamp}&interval=1m&api_token={api_token}&fmt=json"

    df = None
    for attempt in range(5):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if not data:
                print("API returned empty data. Check API key, stock code, and date range. Skipping this period.")
                break

            df = pd.DataFrame(data)
            break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}, retrying (attempt {attempt + 1}/5)...")
            time.sleep(2)

    if df is not None:
        all_data.append(df)
    else:
        print(f"Failed to download data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} even after retries!")

    current_start += interval

if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
    combined_data = combined_data.drop_duplicates(subset='datetime', keep='first')
    combined_data = combined_data.set_index('datetime')

    complete_date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
    combined_data = combined_data.reindex(complete_date_range)

    missing_data = combined_data.isnull().sum()
    if missing_data.any():
        print("\nMissing data report:")
        print(missing_data)
        print("Filling missing values using forward fill...")
        combined_data.ffill(inplace=True)
        print("Missing values filled.")
    else:
        print("\nNo missing data!")

    combined_data = combined_data.reset_index().rename(columns={"index": "datetime"})

    print("Data download complete!")
    #file_name = f"{crypt_ticker}_usd_data.csv"
    file_name = os.path.join("db", f"{crypt_ticker}_usd_data.csv")
    combined_data.to_csv(file_name, index=False)
else:
    print("Data download failed!")