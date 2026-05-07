# BEVAI - Beverage Demand Forecasting Engine

BEVAI is a high-performance, multi-model AI forecasting system designed to predict regional beverage consumption volume. It leverages state-of-the-art machine learning and deep learning architectures to provide real-time market intelligence through an intuitive web dashboard.



## 🚀 Features

- **Multi-Model Support**: Compare projections from XGBoost, LSTM, Transformer, and Facebook Prophet.
- **Automated Recursive Forecasting**: Intelligent logic that handles historical lags and seasonal features automatically.
- **Real-Time API**: Flask-based backend serving model inferences in milliseconds.
- **Clean Dashboard**: A minimalist, high-tech interface for region-specific demand analysis.
- **Performance Optimized**: Automatically identifies and recommends the best-performing model based on Mean Absolute Error (MAE).

## 🧠 Model Architectures

The system evaluates four distinct architectures to ensure the highest accuracy:

1. **XGBoost Regressor (Best Performer)**: Optimized gradient boosting that handles non-linear trends and feature interactions exceptionally well.
2. **LSTM (Deep Learning)**: Long Short-Term Memory network implemented in PyTorch, ideal for capturing long-range dependencies in time series.
3. **Transformer**: A modern sequence model using Attention mechanisms to weigh historical data points differently.
4. **Facebook Prophet**: An additive model for forecasting time series data where non-linear trends are fit with yearly, weekly, and daily seasonality.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, PyTorch, XGBoost, Prophet, Scikit-learn.
- **Frontend**: HTML5, Tailwind CSS (Vanilla Logic), JavaScript.
- **Data Handling**: Pandas, NumPy.
- **Environment**: Virtualized `torch-env` for dependency isolation.

## 📦 Installation & Setup

### 1. Environment Setup
Ensure you have Python 3.10+ installed. Create and activate the virtual environment:

```bash
python -m venv torch-env
# Windows
.\torch-env\Scripts\activate
# Linux/Mac
source torch-env/bin/activate
```

### 2. Install Dependencies
```bash
pip install flask xgboost prophet torch scikit-learn pandas holidays requests
```

### 3. Model Training
To train all models and generate weights for the API:
```bash
python train_models.py
```
This will create a `models/` directory with `.json`, `.pth`, and `.pkl` artifacts.

### 4. Running the API
Start the Flask server:
```bash
python app.py
```
The dashboard will be available at `http://127.0.0.1:5000`.

## 📡 API Reference

### GET `/models`
Returns a list of available AI models and their performance metrics (MAE).

### POST `/forecast`
Generates a multi-week demand projection.
- **Body**: 
  ```json
  {
    "state": "California",
    "steps": 4,
    "model": "XGBoost"
  }
  ```

## 📊 Performance Comparison
Model results are logged in `model_per.csv`. In recent benchmarks, the architectures performed as follows (MAE):
- **XGBoost**: ~6.5M (Winner)
- **LSTM**: ~11.2M
- **Transformer**: ~17.8M
- **Prophet**: ~201M

---

