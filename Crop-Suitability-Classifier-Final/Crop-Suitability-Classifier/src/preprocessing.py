import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]
TARGET = "label"

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Canonicalize common dataset spellings while keeping the model feature
    # names consistent with the capstone specification.
    rename_map = {
        "n": "N",
        "p": "P",
        "k": "K",
        "temperature": "temperature",
        "humidity": "humidity",
        "ph": "ph",
        "rainfall": "rainfall",
        "label": "label",
    }
    df = df.rename(columns=rename_map)

    required = set(FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df[FEATURES + [TARGET]].copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)

    return df, duplicates_removed

def build_numeric_preprocessor(scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
