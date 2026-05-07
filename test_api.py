import requests
import json

url = "http://127.0.0.1:5000/forecast"
data = {
    "state": "California",
    "steps": 4
}

print(f"Requesting forecast for {data['state']} for {data['steps']} weeks...")
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"\nForecast for {result['state']}:")
    for item in result['forecast']:
        print(f"Date: {item['date']}, Predicted Total: {item['predicted_total']:,.2f}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
