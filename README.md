# King County House Price Predictor (Full-Stack ML App)

A small web app with two machine learning models, both trained on the King County, WA house-sales dataset
(`data/kc_house_data.csv`, 21,612 sales from 2014-2015):

- **Regression** (Linear Regression) - predicts the exact sale price of a house.
- **Classification** (Logistic Regression) - predicts the house's **price tier**: Affordable, Mid-range or Premium
  (the cheapest third, middle third and most expensive third of all sales).

| Part | Tech | File |
|------|------|------|
| ML models | scikit-learn, joblib | `train_models.py` |
| Backend | Python FastAPI | `main.py` |
| Frontend | HTML / CSS / JavaScript | `static/index.html` |

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. install the libraries
pip install -r requirements.txt

# 3. train both models (creates the .joblib files in /models)
python train_models.py

# 4. start the backend
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser. The backend serves the web page too, so there is nothing else to start.
Interactive API docs are at http://127.0.0.1:8000/docs.

> Run step 3 on your own computer even though the `models/` folder already has trained files: saved models must be
> loaded with the same scikit-learn version that created them.

## API endpoints

| Method | URL | What it does |
|--------|-----|--------------|
| POST | `/predict/regression` | House details in, predicted price (and likely range) out |
| POST | `/predict/classification` | The same house details in, price tier + confidence out |
| GET | `/options` | Zip codes from the CSV and the model scores (used by the page) |
| GET | `/sample` | A random real house from the CSV (powers the "Try a real sale" button) |

Both prediction endpoints take the same 13 inputs (the form fields): `bedrooms`, `bathrooms`, `sqft_living`,
`sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_basement`, `yr_built`, `renovated`, `zipcode`.
Each one matches a column in the CSV (`renovated` is `yr_renovated > 0`), and the limits the API accepts match the
values that appear in the data.

## Results (on houses the models never saw during training)

| Model | Score |
|-------|-------|
| Regression (price) | **MAE ≈ $76,900**, **R² = 0.867**, typical miss about 10% |
| Classification (tier) | **Accuracy 82.9%** on a held-out 20%, **82%** with 5-fold cross-validation |

Tier cut-offs: **Affordable** under $360,000, **Mid-range** $360,000 to $565,000, **Premium** over $565,000.

The price model is a plain `LinearRegression`, improved by predicting the **log** of the price, using the **zip code**
(one-hot encoded), and using the **log** of the size columns. Together these cut the average error from about
$143,000 to about $77,000. The tier model tries Logistic Regression, Random Forest and Gradient Boosting and keeps
whichever scores best in cross-validation.

## Project structure

```
data/kc_house_data.csv    the dataset
common.py                 shared code: file paths, loading the CSV, feature helpers, tier names
train_models.py           trains, scores and saves both models
models/                   saved models (.joblib) and scores (metrics.json)
main.py                   FastAPI backend
static/index.html         the web page
```
