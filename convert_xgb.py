import pickle
import traceback
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost

try:
    with open('models/lgbm_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print("Type of model:", type(xgb_model))
    
    # XGBoost onnxmltools bug workaround: rename feature names to f0, f1...
    booster = xgb_model.get_booster()
    booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]
    
    initial_type = [('float_input', FloatTensorType([None, 20]))]
    onnx_xgb = convert_xgboost(xgb_model, initial_types=initial_type, target_opset=12)
    with open("api/models/loan.onnx", "wb") as f:
        f.write(onnx_xgb.SerializeToString())
    print("Saved loan.onnx with Float!")
        
except Exception as e:
    with open('api/error.txt', 'w') as f:
        f.write(traceback.format_exc())
