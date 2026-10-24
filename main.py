"""FastAPI backend for the King County house models.

Start it with:   uvicorn main:app --reload
Then open:       http://127.0.0.1:8000
"""
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common import (BASE_DIR, HOUSE_INPUTS, HOUSE_MODEL_PATH, METRICS_PATH, TIER_LABELS,
                    TIER_MODEL_PATH, load_house_data)

app = FastAPI(title="King County house price predictor")

# Lets the page talk to the API even if it is opened from another address.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- Load everything once, when the server starts ----------
if not HOUSE_MODEL_PATH.exists() or not TIER_MODEL_PATH.exists():
    raise SystemExit("Models not found. Run  python train_models.py  first.")

house_model = joblib.load(HOUSE_MODEL_PATH)      # regression: exact price
tier_model = joblib.load(TIER_MODEL_PATH)        # classification: price tier
metrics = json.loads(METRICS_PATH.read_text())
houses = load_house_data()                       # the CSV, used for the dropdown and "real house" button
zipcodes = sorted(houses["zipcode"].unique())


# ---------- What the website is allowed to send us ----------
# The limits below match the values that actually appear in kc_house_data.csv.
class HouseInput(BaseModel):
    bedrooms: int = Field(ge=0, le=11)
    bathrooms: float = Field(ge=0, le=8)
    sqft_living: int = Field(ge=250, le=14000, description="Inside living area in sq ft")
    sqft_lot: int = Field(ge=500, le=1_700_000, description="Land size in sq ft")
    floors: float = Field(ge=1, le=3.5)
    waterfront: int = Field(ge=0, le=1)
    view: int = Field(ge=0, le=4)
    condition: int = Field(ge=1, le=5)
    grade: int = Field(ge=1, le=13)
    sqft_basement: int = Field(ge=0, le=5000)
    yr_built: int = Field(ge=1900, le=2015)
    renovated: int = Field(ge=0, le=1)
    zipcode: str


def house_to_row(house: HouseInput):
    """Check the house, then turn it into a one-row table with the same columns as training."""
    if house.zipcode not in zipcodes:
        raise HTTPException(status_code=422, detail="Please choose a zip code from the list.")
    if house.sqft_basement > house.sqft_living:
        raise HTTPException(status_code=422, detail="The basement can't be bigger than the living area.")
    return pd.DataFrame([house.model_dump()])[HOUSE_INPUTS]


# ---------- The two required endpoints ----------
@app.post("/predict/regression")
def predict_regression(house: HouseInput):
    """House details in, predicted price (and likely range) out."""
    price = float(house_model.predict(house_to_row(house))[0])

    # In testing, 8 out of 10 houses sold within this percentage of the estimate.
    margin = metrics["regression"]["within_80_percent"] / 100
    return {
        "predicted_price": round(price, -3),
        "low": round(price * (1 - margin), -3),
        "high": round(price * (1 + margin), -3),
        "range_percent": metrics["regression"]["within_80_percent"],
    }


@app.post("/predict/classification")
def predict_classification(house: HouseInput):
    """House details in, price tier (Affordable / Mid-range / Premium) + confidence out."""
    chances = tier_model.predict_proba(house_to_row(house))[0]
    by_name = {str(name): float(p) for name, p in zip(tier_model.classes_, chances)}
    best = max(by_name, key=by_name.get)
    return {
        "tier": best,
        "confidence": round(by_name[best], 3),
        "probabilities": {name: round(by_name[name], 3) for name in TIER_LABELS},   # in cheap-to-expensive order
    }


# ---------- Extra endpoints that connect the website to the dataset ----------
@app.get("/options")
def options():
    """Zip codes found in the CSV + the accuracy scores from training."""
    return {"zipcodes": zipcodes, "metrics": metrics}


@app.get("/sample")
def sample():
    """A random real house sale from the CSV, so people can try the models instantly."""
    row = houses.sample(1).iloc[0]
    inputs = {name: row[name] for name in HOUSE_INPUTS}
    inputs = {k: (v if isinstance(v, str) else float(v)) for k, v in inputs.items()}
    for whole_number in ["bedrooms", "sqft_living", "sqft_lot", "waterfront", "view", "condition",
                         "grade", "sqft_basement", "yr_built", "renovated"]:
        inputs[whole_number] = int(inputs[whole_number])
    return {"inputs": inputs, "actual_price": int(row["price"])}


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")
