from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from datetime import timedelta

app = Flask(__name__)

# --- Model Architectures ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x

# --- Loading Logic ---
models = {}
scalers = {}
performance = {}

if os.path.exists('models/xgb_model.json'):
    m = xgb.XGBRegressor()
    m.load_model('models/xgb_model.json')
    models['XGBoost'] = m

if os.path.exists('models/prophet_model.pkl'):
    with open('models/prophet_model.pkl', 'rb') as f:
        models['Prophet'] = pickle.load(f)

if os.path.exists('models/lstm_model.pth'):
    m = LSTMModel(8, 64, 2, 1)
    m.load_state_dict(torch.load('models/lstm_model.pth'))
    m.eval()
    models['LSTM'] = m

if os.path.exists('models/transformer_model.pth'):
    m = TransformerModel(8, 64, 8, 2, 1)
    m.load_state_dict(torch.load('models/transformer_model.pth'))
    m.eval()
    models['Transformer'] = m

for s in ['scaler_X', 'scaler_y']:
    if os.path.exists(f'models/{s}.pkl'):
        with open(f'models/{s}.pkl', 'rb') as f:
            scalers[s] = pickle.load(f)

if os.path.exists('model_per.csv'):
    perf_df = pd.read_csv('model_per.csv')
    performance = dict(zip(perf_df['model'], perf_df['mae']))

data_path = 'cleaned_data.csv'
if os.path.exists(data_path):
    df_full = pd.read_csv(data_path)
    df_full['Date'] = pd.to_datetime(df_full['Date'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    available = []
    best_mae = float('inf')
    best_model = ""
    for m_name in models.keys():
        mae = performance.get(m_name, 0)
        if mae < best_mae and mae > 0:
            best_mae = mae
            best_model = m_name
        available.append({"name": m_name, "mae": mae})
    return jsonify({"models": available, "best": best_model})

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        state = data.get('state')
        steps = int(data.get('steps', 4))
        if steps > 5: steps = 5 # Limit output to 5
        model_name = data.get('model', 'XGBoost')
        
        if model_name not in models:
            return jsonify({"error": f"Model {model_name} not available"}), 400
        
        state_data = df_full[df_full['State'] == state].sort_values('Date')
        last_date = state_data.iloc[-1]['Date']
        history = state_data['Total'].tolist()
        
        forecast_results = []
        for i in range(steps):
            next_date = last_date + timedelta(weeks=1)
            lag_1 = history[-1]
            lag_7 = history[-7] if len(history) >= 7 else history[0]
            lag_30 = history[-30] if len(history) >= 30 else history[0]
            features_raw = [lag_1, lag_7, lag_30, next_date.month, next_date.isocalendar()[1], (next_date.month - 1) // 3 + 1, next_date.year, 0]
            
            if model_name == 'XGBoost':
                df_feat = pd.DataFrame([features_raw], columns=['lag_1', 'lag_7', 'lag_30', 'month', 'week', 'quarter', 'year', 'is_holiday'])
                pred = float(models['XGBoost'].predict(df_feat)[0])
            elif model_name == 'Prophet':
                p_df = pd.DataFrame([{'ds': next_date, 'lag_1': lag_1, 'lag_7': lag_7, 'lag_30': lag_30, 'month': features_raw[3], 'week': features_raw[4], 'quarter': features_raw[5], 'year': features_raw[6], 'is_holiday': 0}])
                f_out = models['Prophet'].predict(p_df)
                pred = float(f_out.iloc[0]['yhat'])
            elif model_name in ['LSTM', 'Transformer']:
                feat_scaled = scalers['scaler_X'].transform([features_raw])
                x_input = torch.tensor([ [feat_scaled[0]] * 8 ], dtype=torch.float32)
                with torch.no_grad():
                    out_scaled = models[model_name](x_input)
                pred = float(scalers['scaler_y'].inverse_transform(out_scaled.numpy())[0,0])

            history.append(pred)
            last_date = next_date
            forecast_results.append({"date": next_date.strftime('%Y-%m-%d'), "predicted_total": pred})
        return jsonify({"state": state, "model": model_name, "forecast": forecast_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Start with debug=True to ensure templates reload
    app.run(host='0.0.0.0', port=5000, debug=True)
