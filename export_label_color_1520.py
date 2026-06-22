import os
import pandas as pd

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

df = pd.read_csv("data/1520/data.csv")
y = df["label"] if "label" in df.columns else df.iloc[:, -1]
labels = sorted(y.astype(str).unique().tolist())

mapping = pd.DataFrame({
    "label": labels,
    "color": [colors[i % len(colors)] for i in range(len(labels))],
})

out_path = "result/visualization_1520/label_color_mapping_1520.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
mapping.to_csv(out_path, index=False)
print(mapping.to_string(index=False))
print("saved:", out_path)
