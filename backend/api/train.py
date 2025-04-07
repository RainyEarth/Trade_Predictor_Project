# api/train.py

import os
import joblib
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from utils.preprocess import prepare_features

router = APIRouter()

@router.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        X, y = prepare_features(df)

        splits = list(TimeSeriesSplit(n_splits=5).split(X))
        train_index, test_index = splits[-1]
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]

        clf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(clf, {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}, cv=3)
        grid.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(grid.best_estimator_, "models/model.pkl")

        return {"message": "Model trained and saved", "best_params": grid.best_params_}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
