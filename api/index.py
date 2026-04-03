from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(title="Medical and Loan Prediction API (ONNX-Powered)", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to ONNX models instead of heavy pickles
LOAN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.onnx')
HEART_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.onnx')

loan_session = None
heart_session = None
stock_session = None

STOCK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stock', 'model.onnx')

try:
    if os.path.exists(LOAN_MODEL_PATH):
        loan_session = ort.InferenceSession(LOAN_MODEL_PATH)
except Exception as e:
    print(f"Error loading ONNX loan model: {e}")

try:
    if os.path.exists(HEART_MODEL_PATH):
        heart_session = ort.InferenceSession(HEART_MODEL_PATH)
except Exception as e:
    print(f"Error loading ONNX heart disease model: {e}")

try:
    if os.path.exists(STOCK_MODEL_PATH):
        stock_session = ort.InferenceSession(STOCK_MODEL_PATH)
except Exception as e:
    print(f"Error loading ONNX stock model: {e}")

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

class StockDataPoint(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float

class StockPriceInput(BaseModel):
    data: list[StockDataPoint] # Expecting exactly 20 points

@app.post("/predict_loan")
def predict_loan(data: LoanPredictionInput):
    if loan_session is None:
        raise HTTPException(status_code=500, detail="Loan ONNX model not loaded")
    
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

    features = np.array([[input_dict[col] for col in expected_cols]], dtype=np.float32)
    
    try:
        # Use the actual names from the model: 'input' and 'label'
        prediction = loan_session.run(['label'], {'input': features})[0][0]
        prediction_label = "Approved" if prediction == 1 else "Rejected"
        
        return {"prediction": int(prediction), "status": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ONNX Prediction error: {str(e)}")

@app.post("/predict_heart_disease")
def predict_heart_disease(data: HeartDiseaseInput):
    if heart_session is None:
        raise HTTPException(status_code=500, detail="Heart Disease ONNX model not loaded")

    try:
        trestbps_log = np.log(data.trestbps) if data.trestbps > 0 else 0
        chol_log = np.log(data.chol) if data.chol > 0 else 0
        thalach_log = np.log(data.thalach) if data.thalach > 0 else 0
        
        features = np.array([[
            data.age, data.sex, data.cp, trestbps_log, chol_log, data.fbs, 
            data.restecg, thalach_log, data.exang, data.oldpeak, data.slope, 
            data.ca, data.thal
        ]], dtype=np.float32)
        
        # Use the actual names from the model: 'float_input' and 'output_label'
        # The model has 2 outputs, typically [label, probabilities]
        outputs = heart_session.run(None, {'float_input': features})
        prediction = outputs[0][0]
        prediction_label = "Heart Disease Present" if prediction == 1 else "No Heart Disease"
        
        response = {"prediction": int(prediction), "status": prediction_label}
        
        # If there's a second output with probabilities
        if len(outputs) > 1:
            prob_output = outputs[1]
            if isinstance(prob_output, list) and len(prob_output) > 0:
                prob_dict = prob_output[0]
                if 0 in prob_dict and 1 in prob_dict:
                    response["probability"] = {
                        "No Disease": float(prob_dict[0]),
                        "Disease": float(prob_dict[1])
                    }
            elif isinstance(prob_output, np.ndarray) and prob_output.shape[1] >= 2:
                response["probability"] = {
                    "No Disease": float(prob_output[0, 0]),
                    "Disease": float(prob_output[0, 1])
                }

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ONNX Prediction error: {str(e)}")

@app.post("/predict_stock_price")
def predict_stock_price(input_data: StockPriceInput):
    if stock_session is None:
        raise HTTPException(status_code=500, detail="Stock ONNX model not loaded")
    
    if len(input_data.data) != 20:
        raise HTTPException(status_code=400, detail="Exactly 20 data points are required")

    # Scaler parameters from research (AAPL 2020-2023)
    MIN_VALS = np.array([-0.445872959091612, -0.44596566901222867, -0.4146727637730872, -0.45386920477129715, 2.5554918670193586e-09, -0.4308479860473215])
    SCALE_VALS = np.array([0.008090593638048678, 0.008077401763183577, 0.008092026829110599, 0.008074094760290541, 1.0967396695276356e-08, 0.008080278853724128])

    try:
        # Convert Pydantic points to numpy array
        raw_data = np.array([[p.open, p.high, p.low, p.close, p.volume, p.average] for p in input_data.data])
        
        # Scaling: (raw - internal_min) * internal_scale + internal_min * internal_scale (Wait, MinMaxScaler: X_scaled = X * scale + min_)
        # Actually, MinMaxScaler fit results in: min_ = -X_min * scale. So: X_scaled = X * scale + min_
        scaled_data = raw_data * SCALE_VALS + MIN_VALS
        
        # Reshape for LSTM: (1, 20, 6)
        input_tensor = scaled_data.reshape(1, 20, 6).astype(np.float32)
        
        # ONNX Inference
        outputs = stock_session.run(['output'], {'input': input_tensor})
        # If output is a scalar (0-dimensional array), convert to float
        prediction_scaled = float(outputs[0])
        
        # Prediction is on the 'Close' price (index 3)
        # Inverse transform: pred_inv = (pred_scaled - min_[3]) / scale_[3]
        prediction_actual = (prediction_scaled - MIN_VALS[3]) / SCALE_VALS[3]
        
        return {
            "predicted_close_price": float(prediction_actual)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stock Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "loan_model_loaded": loan_session is not None,
        "heart_model_loaded": heart_session is not None,
        "stock_model_loaded": stock_session is not None
    }
