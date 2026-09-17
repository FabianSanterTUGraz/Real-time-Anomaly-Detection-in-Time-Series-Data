import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

for root, dirs, files in os.walk(SCRIPT_DIR):
    for filename in files:
        if filename.startswith("acc_") and filename.endswith(".csv"):
            csv_path = os.path.join(root, filename)
            df = pd.read_csv(csv_path, header=None, sep=',')
            if df.shape[1] <= 1:
                print(f"Skipped (already reduced): {csv_path}")
                continue
            df = df.iloc[:, [3]]
            df.to_csv(csv_path, header=False, index=False)
            print(f"Reduced: {csv_path}")
