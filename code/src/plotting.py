import numpy as np
import pandas as pd
from matplotlib.pyplot import colormaps
import matplotlib.pyplot as plt

file_path = r'C:\Users\39320\Desktop\bachelor_thesis\Bachelor_Thesis_Code\Implementation\Data\output.txt'

# MODIFIED: Use read_csv and specify the comma delimiter to unpack columns
# names=['X', 'Y'] assigns clean column names automatically
df = pd.read_csv(file_path, nrows= 30000, header=None, sep=',', names=['X', 'Y'])

print("First few rows of extracted coordinates:")
print(df.head())

projected = df[['X', 'Y']].to_numpy()

# Color by distance from the origin, same as the reference implementation's
# "scores" (norm of each projected point), normalized to [0, 1].
scores = np.linalg.norm(projected, axis=1)
scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# --- 2D VISUAL FINGERPRINT PLOT ---
plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(projected[:, 0], projected[:, 1], s=8, c=colormaps["turbo"](scores_norm))
ax.axis("off")
ax.set_title("TDE", fontsize=30)

# Save the fingerprint layout graph
plt.savefig("tde.png")
plt.show()