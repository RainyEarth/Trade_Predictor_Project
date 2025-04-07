import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.fft import fft
from pywt import wavedec
from pykalman import KalmanFilter

def add_technical_indicators(df):
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Bollinger_Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_fft_features(close_prices):
    fft_vals = np.abs(fft(close_prices))[:10]
    return pd.DataFrame([fft_vals], columns=[f'fft_{i}' for i in range(len(fft_vals))])

def compute_wavelet_features(series):
    coeffs = wavedec(series, 'db1', level=3)
    features = []
    for i, coeff in enumerate(coeffs):
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
    return pd.DataFrame([features], columns=[f'wavelet_{i}' for i in range(len(features))])

def apply_kalman_filter(series):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
    state_means, _ = kf.filter(series.values.reshape(-1, 1))
    return pd.Series(state_means.flatten(), index=series.index)

def compute_tda_features(df):
    df['tda_dummy'] = df['Close'].rolling(window=10).mean()  # Placeholder for actual TDA
    return df

def prepare_features(df):
    df = df.copy()
    df = add_technical_indicators(df)
    df['Kalman_Close'] = apply_kalman_filter(df['Close'])

    # Add ARIMA and Markov Regime if needed here (like in main.py)

    # Combine FFT + Wavelet
    fft_df = compute_fft_features(df['Close'])
    wavelet_df = compute_wavelet_features(df['Close'])

    df = pd.concat([df.reset_index(drop=True), fft_df, wavelet_df], axis=1)

    df = compute_tda_features(df)

    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA', 'MACD', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower',
        'Kalman_Close'
    ] + [col for col in df.columns if col.startswith('fft_') or col.startswith('wavelet_') or col.startswith('tda_')]

    X = df[features]
    y = df['Target']

    # Optional PCA
    # pca = PCA(n_components=10)
    # X = pd.DataFrame(pca.fit_transform(X))

    return X, y
