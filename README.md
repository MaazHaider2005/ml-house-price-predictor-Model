# House Price Prediction App

A machine learning web application that predicts house prices and classifies house categories using trained models.

## Features

- Predict house prices using a trained machine learning model.
- Classify house categories.
- FastAPI backend.
- HTML frontend.
- Saved models and evaluation metrics.

## Project Structure

- `main.py` — application backend
- `common.py` — shared helper functions
- `train_models.py` — model training script
- `data/` — dataset
- `models/` — trained models and metrics
- `static/index.html` — web interface
- `EXPLAIN.md` — project explanation
- `requirements.txt` — Python dependencies

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Models

```bash
python train_models.py
```

## Run the Application

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## License

For educational purposes.
