# Medical and Loan Prediction API Documentation

This document describes the expected input format and output format for the deployed FastAPI models.

---

## 1. Loan Prediction API

**Endpoint**: `/predict_loan`  
**Method**: `POST`

### Expected Input (JSON)
The API accepts a JSON object with the following fields:

| Field | Type | Example / Allowed Values | Description |
|-------|------|--------------------------|-------------|
| `Gender` | String | `"Male"`, `"Female"` | Applicant's gender |
| `Married` | String | `"Yes"`, `"No"` | Marital status |
| `Dependents` | String | `"0"`, `"1"`, `"2"`, `"3+"` | Number of dependents |
| `Education` | String | `"Graduate"`, `"Not Graduate"` | Education level |
| `Self_Employed`| String | `"Yes"`, `"No"` | Self employment status |
| `ApplicantIncome` | Float | `5849.0` | Applicant's monthly income |
| `CoapplicantIncome`| Float | `0.0` | Co-applicant's monthly income |
| `LoanAmount` | Float | `146.0` | Loan amount requested |
| `Loan_Amount_Term`| Float | `360.0` | Term of the loan in months |
| `Credit_History` | Float | `1.0`, `0.0` | Credit history meets guidelines (1.0 = Yes) |
| `Property_Area`| String | `"Urban"`, `"Rural"`, `"Semiurban"`| Area of property |

### Example Input Payload:
```json
{
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
```

### Expected Output
The API returns a JSON object representing the prediction decision:
```json
{
  "prediction": 1,
  "status": "Approved"
}
```
- `prediction`: Integer (`1` for Approval, `0` for Rejection)
- `status`: String (`"Approved"` or `"Rejected"`)

---

## 2. Heart Disease Prediction API

**Endpoint**: `/predict_heart_disease`  
**Method**: `POST`

### Expected Input (JSON)
The API accepts a JSON object with the following medical features:

| Field | Type | Description |
|-------|------|-------------|
| `age` | Float | Age in years |
| `sex` | Integer | Sex (1 = male; 0 = female) |
| `cp` | Integer | Chest pain type (0, 1, 2, 3) |
| `trestbps` | Float | Resting blood pressure (in mm Hg) |
| `chol` | Float | Serum cholestoral in mg/dl |
| `fbs` | Integer | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg` | Integer | Resting electrocardiographic results (0, 1, 2) |
| `thalach` | Float | Maximum heart rate achieved |
| `exang` | Integer | Exercise induced angina (1 = yes; 0 = no) |
| `oldpeak` | Float | ST depression induced by exercise relative to rest |
| `slope` | Integer | The slope of the peak exercise ST segment (0, 1, 2) |
| `ca` | Integer | Number of major vessels (0-4) colored by flourosopy |
| `thal` | Integer | Thalassemia (0, 1, 2, 3) |

### Example Input Payload:
```json
{
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
```

### Expected Output
The API returns a JSON object containing the prediction and probability:
```json
{
  "prediction": 1,
  "status": "Heart Disease Present",
  "probability": {
    "No Disease": 0.2287,
    "Disease": 0.7712
  }
}
```
- `prediction`: Integer (`1` for Presence, `0` for Absence)
- `status`: String interpretation (`"Heart Disease Present"` or `"No Heart Disease"`)
- `probability`: Float dictionary displaying confidence.

*(Note: The API automatically handles necessary log transformations for `trestbps`, `chol`, and `thalach` internally before feeding it to the ONNX/Pickle model)*

---

## 3. Stock Price Prediction API

**Endpoint**: `/predict_stock_price`  
**Method**: `POST`

### Expected Input (JSON)
The API accepts a JSON object containing a list of exactly **20 historical data points**.

| Field | Type | Description |
|-------|------|-------------|
| `open` | Float | Opening price of the day |
| `high` | Float | Highest price of the day |
| `low` | Float | Lowest price of the day |
| `close` | Float | Closing price of the day |
| `volume`| Float | Trading volume |
| `average`| Float | Average price ((High + Low) / 2) |

### Example Input Payload:
```json
{
  "data": [
    {
      "open": 150.0,
      "high": 155.0,
      "low": 149.0,
      "close": 152.0,
      "volume": 1200000.0,
      "average": 152.0
    },
    ... (repeated for 20 unique days)
  ]
}
```

### Expected Output
The API returns the predicted closing price for the next day:
```json
{
  "predicted_close_price": 153.45
}
```

*(Note: The API automatically handles internal Min-Max scaling based on the training dataset (AAPL 2020-2023) before processing the sequence through the LSTM model)*
