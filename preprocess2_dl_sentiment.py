import requests
import pandas as pd
from datetime import datetime
import yaml
import os

with open("preprocess.yaml", "r") as f:
    config = yaml.safe_load(f)
crypt_ticker = config["data_download"]["crypt_ticker"]

api_token = "demo"

start_date = "2021-09-17"
end_date = datetime.now().strftime("%Y-%m-%d")

url = f'https://eodhd.com/api/sentiments?s={crypt_ticker}-usd.cc&from={start_date}&to={end_date}&api_token={api_token}&fmt=json'

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    try:
        ticker = list(data.keys())[0]
        sentiments = data[ticker]
    except IndexError:
        print("API returned empty data. Check API key, stock code, and date range.")
        exit(1)

    df = pd.DataFrame(sentiments)
    df['date'] = pd.to_datetime(df['date'])
    date_range = pd.date_range(start=start_date, end=end_date)
    df = df.set_index('date')
    df = df.reindex(date_range)

    missing_data = df.isnull().sum()
    if missing_data.any():
        print("\nMissing data report:")
        print(missing_data)
        print("Filling missing values using forward fill...")
        df.ffill(inplace=True)
        print("Missing values filled.")
    else:
        print("\nNo missing data!")

    df = df.reset_index().rename(columns={"index": "datetime"})

    file_name = os.path.join("db", f"{crypt_ticker}_sentiments.csv")
    df.to_csv(file_name, index=False)
    print(f"Successfully saved file: {file_name}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")