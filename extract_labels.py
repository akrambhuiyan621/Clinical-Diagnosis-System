# extract_labels.py
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Change "your_dataset.csv" and "disease" to match your actual file and column name
df = pd.read_csv("your_dataset.csv")
print("Columns:", df.columns.tolist())
print("Unique diseases:", df["disease"].unique())  # change "disease" to your label column name

le = LabelEncoder()
le.fit(df["disease"])  # change "disease" to your label column name

print(f"\nFound {len(le.classes_)} classes:")
for i, label in enumerate(le.classes_):
    print(f"  {i}: {label}")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nlabel_encoder.pkl rebuilt with real labels!")