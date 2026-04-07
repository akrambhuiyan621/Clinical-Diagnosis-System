# find_labels.py
import pandas as pd

# Try common dataset filenames - adjust if yours is different
import os
for f in os.listdir("."):
    if f.endswith(".csv") or f.endswith(".xlsx") or f.endswith(".json"):
        print("Found:", f)
        