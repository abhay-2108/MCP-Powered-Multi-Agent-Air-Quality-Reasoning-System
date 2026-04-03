!pip install yfinance
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length, 3]
        return x, y
def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['Average'] = (df['High'] + df['Low']) / 2
    return df
def split_scale(df):
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    train = df[:train_size]
    val = df[train_size:train_size+val_size]
    test = df[train_size+val_size:]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()
def train_model(model, train_loader, val_loader, epochs=20, save_path="best_model.pth"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss={val_loss:.6f}")
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    return train_losses, val_losses
def evaluate(model, test_loader, scaler):
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.extend(pred)
            actuals.extend(y.numpy())
    preds = np.array(preds)
    actuals = np.array(actuals)
    dummy_pred = np.zeros((len(preds), 6))
    dummy_actual = np.zeros((len(actuals), 6))
    dummy_pred[:, 3] = preds
    dummy_actual[:, 3] = actuals
    preds_inv = scaler.inverse_transform(dummy_pred)[:, 3]
    actuals_inv = scaler.inverse_transform(dummy_actual)[:, 3]
    mse = mean_squared_error(actuals_inv, preds_inv)
    mae = mean_absolute_error(actuals_inv, preds_inv)
    r2 = r2_score(actuals_inv, preds_inv)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    plt.figure()
    plt.plot(actuals_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted')
    plt.legend()
    plt.grid(True)
    plt.title("Prediction vs Actual")
    plt.show()
symbol = 'AAPL'
df = fetch_data(symbol, '2020-01-01', '2023-01-01')
train, val, test, scaler = split_scale(df)
seq_length = 20
train_ds = StockDataset(torch.FloatTensor(train), seq_length)
val_ds = StockDataset(torch.FloatTensor(val), seq_length)
test_ds = StockDataset(torch.FloatTensor(test), seq_length)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)
model = LSTMModel().to(device)
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=150)
evaluate(model, test_loader, scaler)
!pip install onnxscript
def export_onnx(model, seq_length=20, path="model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, seq_length, 6).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"Model exported to ONNX format at {path}")
export_onnx(model)
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.grid(True)
plt.title("Training vs Validation Loss")
plt.show()