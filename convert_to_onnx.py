import pickle
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import lightgbm as lgb
from onnxmltools import convert_lightgbm

print("Converting Random Forest...")
try:
    with open('models/heart.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Random forest has 13 features
    initial_type = [('float_input', FloatTensorType([None, 13]))]
    onnx_rf = convert_sklearn(rf_model, initial_types=initial_type)
    with open("api/models/heart.onnx", "wb") as f:
        f.write(onnx_rf.SerializeToString())
    print("Saved heart.onnx")
except Exception as e:
    print(f"Error converting Random Forest: {e}")

print("Converting LightGBM...")
try:
    with open('models/lgbm_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    
    # LightGBM has 20 features
    initial_type = [('float_input', FloatTensorType([None, 20]))]
    # Set the target_opset safely
    onnx_lgb = convert_lightgbm(lgb_model, initial_types=initial_type, target_opset=12)
    with open("api/models/loan.onnx", "wb") as f:
        f.write(onnx_lgb.SerializeToString())
    print("Saved loan.onnx")
except Exception as e:
    print(f"Error converting LightGBM: {e}")

print("Done!")
