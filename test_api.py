import requests
import json

BASE_URL = "https://mcp-powered-multi-agent-air-quality.vercel.app" 

print("--- Testing API Health ---")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
except Exception as e:
    print(f"Failed to connect: {e}\n")

print("--- Testing Loan Prediction API ---")
loan_data = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 146.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}
try:
    response = requests.post(f"{BASE_URL}/predict_loan", json=loan_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
except Exception as e:
    print(f"Failed to predict loan: {e}\n")

print("--- Testing Heart Disease Prediction API ---")
heart_data = {
    "age": 63.0,
    "sex": 1,
    "cp": 3,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}
try:
    response = requests.post(f"{BASE_URL}/predict_heart_disease", json=heart_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
except Exception as e:
    print(f"Failed to predict heart disease: {e}\n")
