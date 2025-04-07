# api/predict.py

import os
import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.preprocess import prepare_features

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not os.path.exists("models/model.pkl"):
            raise HTTPException(status_code=404, detail="Model not found. Train first.")

        model = joblib.load("models/model.pkl")
        df = pd.read_csv(file.file)
        X, y = prepare_features(df)
        y_pred = model.predict(X)

        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

chart_data = df[['Close', 'ARIMA_Pred']].copy()
chart_data['index'] = chart_data.index
chart_data = chart_data.tail(100)  # Optional: limit to last 100 points

return {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "chart_data": chart_data.to_dict(orient="records")
}
