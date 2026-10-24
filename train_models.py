"""Train both house models, print their scores, and save them with joblib.

  1. REGRESSION      - predict the exact sale price (in dollars)
  2. CLASSIFICATION  - predict the price tier: Affordable / Mid-range / Premium

Run it with:   python train_models.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common import (HOUSE_INPUTS, HOUSE_MODEL_PATH, METRICS_PATH, MODELS_DIR, TIER_LABELS,
                    TIER_MODEL_PATH, add_features, build_preprocessor, load_house_data)

MODELS_DIR.mkdir(exist_ok=True)
metrics = {}

houses = load_house_data()
X = houses[HOUSE_INPUTS]          # what we know about the house
y = houses["price"]               # what we want to predict

# Keep 20% of the houses hidden so we can test the models on houses they have never seen.
# (Same split for both models, so their scores are comparable.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# PART 1 - LINEAR REGRESSION: predict the exact price
# =====================================================================
print("=== House price (Linear Regression) ===")

# Step 1: add log-sizes.  Step 2: zip code -> 0/1 columns, keep other numbers.
# Step 3: fit a straight-line (linear) model.
linear_steps = Pipeline([
    ("add_features", FunctionTransformer(add_features)),
    ("prepare", build_preprocessor()),
    ("linear", LinearRegression()),
])

# Predicting the LOG of the price (then converting back) is the biggest accuracy boost:
# it stops expensive houses from dominating the fit.
house_model = TransformedTargetRegressor(regressor=linear_steps, func=np.log, inverse_func=np.exp)
house_model.fit(X_train, y_train)

predictions = house_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
percent_errors = np.abs(predictions - y_test) / y_test * 100

print(f"R2  : {r2:.3f}   (1.0 = perfect)")
print(f"MAE : ${mae:,.0f}   (average miss, in dollars)")
print(f"Typical miss: {np.median(percent_errors):.1f}%   |   8 in 10 houses within {np.percentile(percent_errors, 80):.1f}%")

metrics["regression"] = {
    "r2": round(r2, 3),
    "mae": round(mae),
    "median_error_percent": round(float(np.median(percent_errors)), 1),
    "within_80_percent": round(float(np.percentile(percent_errors, 80)), 1),
    "houses_in_dataset": len(houses),
}

# Scores above come from the hidden 20%. Now re-train on ALL houses for the final saved model.
house_model.fit(X, y)
joblib.dump(house_model, HOUSE_MODEL_PATH)

# =====================================================================
# PART 2 - CLASSIFICATION: predict the price tier
# =====================================================================
print("\n=== Price tier (Classification) ===")

# Turn the price into 3 groups of equal size. The two cut-off prices are rounded to $1,000.
cut_low, cut_high = (round(float(v), -3) for v in np.quantile(y, [1 / 3, 2 / 3]))


def to_tier(prices):
    """Price in dollars -> 'Affordable' / 'Mid-range' / 'Premium'."""
    return pd.Series(np.where(prices < cut_low, TIER_LABELS[0],
                     np.where(prices < cut_high, TIER_LABELS[1], TIER_LABELS[2])), index=prices.index)


tier_train, tier_test, tier_all = to_tier(y_train), to_tier(y_test), to_tier(y)
print(f"Affordable: under ${cut_low:,.0f}   Mid-range: ${cut_low:,.0f} to ${cut_high:,.0f}   Premium: ${cut_high:,.0f}+")


def tier_pipeline(classifier, scale):
    return Pipeline([
        ("add_features", FunctionTransformer(add_features)),
        ("prepare", build_preprocessor(scale=scale)),
        ("classify", classifier),
    ])


# Try three models and keep whichever scores best in cross-validation on the training houses.
candidates = {
    "Logistic Regression": tier_pipeline(LogisticRegression(max_iter=2000), scale=True),
    "Random Forest": tier_pipeline(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42), scale=False),
    "Gradient Boosting": tier_pipeline(GradientBoostingClassifier(random_state=42), scale=False),
}
best_name, best_score = None, 0
for name, candidate in candidates.items():
    score = cross_val_score(candidate, X_train, tier_train, cv=5).mean()
    print(f"  {name:20s} cross-validation accuracy: {score:.3f}")
    if score > best_score:
        best_name, best_score = name, score

tier_model = candidates[best_name]
tier_model.fit(X_train, tier_train)
accuracy = accuracy_score(tier_test, tier_model.predict(X_test))
print(f"Chosen model: {best_name}")
print(f"Accuracy on {len(X_test):,} held-out houses: {accuracy:.1%}")

metrics["classification"] = {
    "accuracy": round(accuracy, 3),
    "cv_accuracy": round(best_score, 3),
    "model": best_name,
    "labels": TIER_LABELS,
    "cut_low": cut_low,
    "cut_high": cut_high,
    "houses_in_dataset": len(houses),
}

tier_model.fit(X, tier_all)               # final model learns from all houses
joblib.dump(tier_model, TIER_MODEL_PATH)

# Save the scores so the website can show them.
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print("\nSaved: house_price_model.joblib, house_tier_model.joblib, metrics.json")
