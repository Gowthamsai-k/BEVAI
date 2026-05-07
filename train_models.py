import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import holidays

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

def save_mod_per(model, val):
    print(f"Saving {model}: MAE={val}")
    res_df = pd.DataFrame({'model': [model], 'mae': [val]})
    res_df.to_csv('model_per.csv', mode='a', index=False, header=False)

# Load data
df = pd.read_csv('cleaned_data.csv')
if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df = df.drop(columns=[df.columns[0]])

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['State', 'Date'])

# Define features
exog_cols = ['lag_1', 'lag_7', 'lag_30', 'month', 'week', 'quarter', 'year', 'is_holiday']
df_clean = df.dropna(subset=['lag_1', 'lag_7', 'lag_30', 'Total'])

# Split data (80/20)
train_size = int(len(df_clean) * 0.8)
train = df_clean.iloc[:train_size]
test = df_clean.iloc[train_size:]

X_train = train[exog_cols]
y_train = train['Total']
X_test = test[exog_cols]
y_test = test['Total']

# 1. XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
save_mod_per('XGBoost', xgb_mae)
xgb_model.save_model('models/xgb_model.json')

# 2. Facebook Prophet
print("Training Prophet...")
prophet_train = train[['Date', 'Total']].rename(columns={'Date': 'ds', 'Total': 'y'})
for col in exog_cols:
    prophet_train[col] = train[col].values

m = Prophet()
for col in exog_cols:
    m.add_regressor(col)

m.fit(prophet_train)

prophet_test = test[['Date']].rename(columns={'Date': 'ds'})
for col in exog_cols:
    prophet_test[col] = test[col].values

forecast = m.predict(prophet_test)
prophet_mae = mean_absolute_error(y_test, forecast['yhat'])
save_mod_per('Prophet', prophet_mae)
with open('models/prophet_model.pkl', 'wb') as f:
    pickle.dump(m, f)

# 3. LSTM (PyTorch)
print("Training LSTM...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size=8):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size
    def __len__(self):
        return len(self.X) - self.window_size
    def __getitem__(self, idx):
        return self.X[idx:idx+self.window_size], self.y[idx+self.window_size]

window_size = 8
train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, window_size)
test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, window_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMModel(len(exog_cols), 64, 2, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

epochs = 10
lstm_model.train()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

torch.save(lstm_model.state_dict(), 'models/lstm_model.pth')
lstm_model.eval()
lstm_preds = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = lstm_model(batch_X)
        lstm_preds.extend(outputs.numpy())

lstm_preds_inv = scaler_y.inverse_transform(np.array(lstm_preds).reshape(-1, 1))
lstm_mae = mean_absolute_error(y_test.values[window_size:], lstm_preds_inv)
save_mod_per('LSTM', lstm_mae)

# 4. Transformers (PyTorch)
print("Training Transformer...")
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

trans_model = TransformerModel(len(exog_cols), d_model=64, nhead=8, num_layers=2, output_size=1)
optimizer = torch.optim.Adam(trans_model.parameters(), lr=0.001)

trans_model.train()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = trans_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

torch.save(trans_model.state_dict(), 'models/transformer_model.pth')
trans_model.eval()
trans_preds = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = trans_model(batch_X)
        trans_preds.extend(outputs.numpy())

trans_preds_inv = scaler_y.inverse_transform(np.array(trans_preds).reshape(-1, 1))
trans_mae = mean_absolute_error(y_test.values[window_size:], trans_preds_inv)
save_mod_per('Transformer', trans_mae)

# Save scalers
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("All models and scalers saved to 'models/' directory.")
