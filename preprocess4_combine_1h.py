import pandas as pd
import yaml
import os

with open("preprocess.yaml", "r") as f:
    config = yaml.safe_load(f)
crypt_ticker = config["data_download"]["crypt_ticker"]

def resample_to_hourly(input_csv, output_csv):
    """
    Resamples minute-level CSV data to hourly data.
    """
    try:
        df = pd.read_csv(input_csv, parse_dates=['datetime'])
        df = df.set_index('datetime')

        hourly_df = df.resample('h').agg({
            'timestamp': 'first',
            'gmtoffset': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'count': 'first',
            'normalized': 'first'
        })

        hourly_df = hourly_df.reset_index()
        hourly_df.to_csv(output_csv, index=False)

        print(f"Hourly data saved to: {output_csv}")

    except FileNotFoundError:
        print(f"Error: File {input_csv} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = os.path.join("db", f"{crypt_ticker}_merged_1m.csv")
output_file = os.path.join("db", f"{crypt_ticker}_merged_1h.csv")

resample_to_hourly(input_file, output_file)