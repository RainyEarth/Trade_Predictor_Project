# FastAPI backend for NeuralAditya trade prediction

import os
import joblib
import logging
import traceback
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from tempfile import NamedTemporaryFile

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "models/model.pkl"

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        tmp = NamedTemporaryFile(delete=False)
        tmp.write(await file.read())
        tmp.close()

        df = pd.read_csv(tmp.name)
        os.remove(tmp.name)

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"CSV missing required columns: {required_cols}")

        df.dropna(inplace=True)

        # Basic features
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # ARIMA
        try:
            arima_model = ARIMA(df['Close'], order=(5, 1, 0)).fit()
            df['ARIMA_Pred'] = arima_model.predict(start=1, end=len(df))
        except Exception:
            logging.warning("ARIMA model failed")
            df['ARIMA_Pred'] = 0

        # Markov Regimes
        try:
            mod = MarkovAutoregression(df['Close'], k_regimes=2, order=1).fit()
            df['Regime'] = mod.smoothed_marginal_probabilities[0]
        except Exception:
            logging.warning("Markov model failed")
            df['Regime'] = 0

        # Features and labels
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ARIMA_Pred', 'Regime']
        df.dropna(inplace=True)
        X, y = df[features], df['Target']

        # Time-based split
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Grid search
        clf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        grid = GridSearchCV(clf, param_grid, cv=3)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed. See server logs for details.")
