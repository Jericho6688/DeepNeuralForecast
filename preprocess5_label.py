import pandas as pd
import numpy as np
import yaml
import os

with open("preprocess.yaml", "r") as f:
    config = yaml.safe_load(f)

crypt_ticker = config["data_label"]["crypt_ticker"]
interval = config["data_label"]["interval"]
future_interval = config["data_label"]["future_interval"]
change_threshold = config["data_label"]["change_threshold"]

filepath = f"{crypt_ticker}_merged_{interval}.csv"

try:
    df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

df['signal'] = 0

df['future_high'] = df['high'].rolling(window=future_interval, min_periods=1).max().shift(-future_interval)
df['future_low'] = df['low'].rolling(window=future_interval, min_periods=1).min().shift(-future_interval)

df['signal'] = np.where(
    (df['future_high'] - df['close']) / df['close'] >= change_threshold,
    1,
    np.where(
        (df['future_low'] - df['close']) / df['close'] <= -change_threshold,
        -1,
        0
    )
)

df.drop(['future_high', 'future_low'], axis=1, inplace=True)

print("Signal calculation complete.")

signal_counts = df['signal'].value_counts()
print("\nSignal Report:")
for signal, count in signal_counts.items():
    print(f"Signal {signal}: {count}")

signal_percentages = df['signal'].value_counts(normalize=True) * 100
print("\nSignal Percentages:")
for signal, percentage in signal_percentages.items():
    print(f"Signal {signal}: {percentage:.2f}%")

print(df.tail(70))

file_name = os.path.join("db", f"{crypt_ticker}_alabel_{interval}.csv")
df.to_csv(file_name, index=True)