import requests
import json
import numpy as np

BASE_URL = "http://127.0.0.1:8001"

print("--- Testing API Health ---")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
except Exception as e:
    print(f"Failed to connect: {e}\n")

print("--- Testing Stock Price Prediction API ---")
# Generate 20 days of dummy data
stock_data = []
base_price = 150.0
for i in range(20):
    open_p = base_price + np.random.randn()
    high_p = open_p + abs(np.random.randn())
    low_p = open_p - abs(np.random.randn())
    close_p = (high_p + low_p) / 2 + np.random.randn()
    volume = 1000000 + np.random.randint(-100000, 100000)
    average = (high_p + low_p) / 2
    stock_data.append({
        "open": round(float(open_p), 2),
        "high": round(float(high_p), 2),
        "low": round(float(low_p), 2),
        "close": round(float(close_p), 2),
        "volume": round(float(volume), 2),
        "average": round(float(average), 2)
    })

payload = {"data": stock_data}

try:
    response = requests.post(f"{BASE_URL}/predict_stock_price", json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    else:
        print(f"Error: {response.text}\n")
except Exception as e:
    print(f"Failed to predict stock price: {e}\n")
