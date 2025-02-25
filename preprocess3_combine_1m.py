import pandas as pd
import yaml
import os

with open("preprocess.yaml", "r") as f:
    config = yaml.safe_load(f)
crypt_ticker = config["data_download"]["crypt_ticker"]

minute_df = pd.read_csv(f"{crypt_ticker}_usd_data.csv")
daily_df = pd.read_csv(f"{crypt_ticker}_sentiments.csv")

minute_df['datetime'] = pd.to_datetime(minute_df['datetime'])
daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])

minute_df['date'] = minute_df['datetime'].dt.strftime('%Y-%m-%d')
daily_df['date'] = daily_df['datetime'].dt.strftime('%Y-%m-%d')

merged_df = pd.merge(minute_df, daily_df[['date', 'count', 'normalized']], on='date', how='left')
merged_df.drop(columns=['date'], inplace=True)

merged_df[['count', 'normalized']] = merged_df[['count', 'normalized']].ffill()

file_name = os.path.join("db", f"{crypt_ticker}_merged_1m.csv")
merged_df.to_csv(file_name, index=False)

print(merged_df)