from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

# Create standard FastAPI app without wrapping it
app = FastAPI(title="Medical and Loan Prediction API", version="1.0")

# Note: Vercel places the executing file exactly here
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the copies placed inside api/models/
LOAN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.pkl')
HEART_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart.pkl')

# Load models
loan_model = None
heart_model = None

try:
    with open(LOAN_MODEL_PATH, 'rb') as f:
        loan_model = pickle.load(f)
except Exception as e:
    print(f"Error loading loan model: {e}")

try:
    with open(HEART_MODEL_PATH, 'rb') as f:
        heart_model = pickle.load(f)
except Exception as e:
    print(f"Error loading heart disease model: {e}")

# Loan Prediction Schema
class LoanPredictionInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

# Heart Disease Prediction Schema
class HeartDiseaseInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict_loan")
def predict_loan(data: LoanPredictionInput):
    if loan_model is None:
        raise HTTPException(status_code=500, detail="Loan model not loaded")
    
    expected_cols = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
        'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes', 'Dependents_0', 'Dependents_1', 
        'Dependents_2', 'Dependents_3+', 'Education_Graduate', 'Education_Not Graduate', 
        'Self_Employed_No', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 
        'Property_Area_Urban'
    ]
    
    input_dict = {col: 0.0 for col in expected_cols}
    input_dict['ApplicantIncome'] = data.ApplicantIncome
    input_dict['CoapplicantIncome'] = data.CoapplicantIncome
    input_dict['LoanAmount'] = data.LoanAmount
    input_dict['Loan_Amount_Term'] = data.Loan_Amount_Term
    input_dict['Credit_History'] = data.Credit_History
    
    if f'Gender_{data.Gender}' in input_dict: input_dict[f'Gender_{data.Gender}'] = 1.0
    if f'Married_{data.Married}' in input_dict: input_dict[f'Married_{data.Married}'] = 1.0
    if f'Dependents_{data.Dependents}' in input_dict: input_dict[f'Dependents_{data.Dependents}'] = 1.0
    if f'Education_{data.Education}' in input_dict: input_dict[f'Education_{data.Education}'] = 1.0
    if f'Self_Employed_{data.Self_Employed}' in input_dict: input_dict[f'Self_Employed_{data.Self_Employed}'] = 1.0
    if f'Property_Area_{data.Property_Area}' in input_dict: input_dict[f'Property_Area_{data.Property_Area}'] = 1.0

    df = pd.DataFrame([input_dict], columns=expected_cols)
    try:
        prediction = loan_model.predict(df)[0]
        prediction_label = "Approved" if prediction == 1 else "Rejected"
        return {"prediction": int(prediction), "status": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_heart_disease")
def predict_heart_disease(data: HeartDiseaseInput):
    if heart_model is None:
        raise HTTPException(status_code=500, detail="Heart Disease model not loaded")

    try:
        trestbps_log = np.log(data.trestbps) if data.trestbps > 0 else 0
        chol_log = np.log(data.chol) if data.chol > 0 else 0
        thalach_log = np.log(data.thalach) if data.thalach > 0 else 0
        
        features = np.array([[
            data.age, data.sex, data.cp, trestbps_log, chol_log, data.fbs, 
            data.restecg, thalach_log, data.exang, data.oldpeak, data.slope, 
            data.ca, data.thal
        ]])
        
        prediction = heart_model.predict(features)[0]
        prediction_label = "Heart Disease Present" if prediction == 1 else "No Heart Disease"
        
        if hasattr(heart_model, "predict_proba"):
            probability = heart_model.predict_proba(features)[0].tolist()
            return {
                "prediction": int(prediction), 
                "status": prediction_label,
                "probability": { "No Disease": probability[0], "Disease": probability[1] }
            }
            
        return {"prediction": int(prediction), "status": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "loan_model_loaded": loan_model is not None,
        "heart_model_loaded": heart_model is not None
    }

# Ensure Vercel can wrap the ASGI app directly
