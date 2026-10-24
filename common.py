"""Shared helpers used by BOTH train_models.py and main.py.

Keeping them in one place means training and the API always prepare
the data in exactly the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- File locations ----------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "kc_house_data.csv"
MODELS_DIR = BASE_DIR / "models"
HOUSE_MODEL_PATH = MODELS_DIR / "house_price_model.joblib"
TIER_MODEL_PATH = MODELS_DIR / "house_tier_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---------- Price tiers for the classification model ----------
# Every sale is put in one of three groups of equal size (cheapest third,
# middle third, most expensive third). The cut-off prices are saved in metrics.json.
TIER_LABELS = ["Affordable", "Mid-range", "Premium"]

# ---------- The 13 things we ask about a house ----------
HOUSE_INPUTS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_basement",
    "yr_built", "renovated", "zipcode",
]


def load_house_data():
    """Read the King County house-sales CSV and tidy it up."""
    df = pd.read_csv(DATA_PATH, dtype={"zipcode": str})
    df = df[df["bedrooms"] < 15]                      # one row says 33 bedrooms (a typo)
    df["floors"] = df["floors"].astype(float)
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)   # 1 = has been renovated
    return df.reset_index(drop=True)


def add_features(df):
    """Add 'log' versions of the size columns.

    House prices grow in percentages, not in fixed amounts (a bigger house
    adds proportionally more value), so log-sizes fit a straight line better.
    """
    df = df.copy()
    df["log_sqft_living"] = np.log(df["sqft_living"])
    df["log_sqft_lot"] = np.log(df["sqft_lot"])
    return df


def build_preprocessor(scale=False):
    """Turn the house table into numbers a model can use (shared by both models).

    - zip code  -> one 0/1 column per zip (it is a label, not a quantity)
    - the rest  -> kept as numbers (optionally scaled, which Logistic Regression likes)
    """
    return ColumnTransformer([
        ("zipcode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["zipcode"]),
        ("numbers", StandardScaler() if scale else "passthrough", [
            "bedrooms", "bathrooms", "log_sqft_living", "log_sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_basement",
            "yr_built", "renovated",
        ]),
    ])
