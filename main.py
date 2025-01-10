from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import ccxt
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

app = FastAPI(title="Cryptocurrency Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

def initialize_binance_client():
    return ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET
    })

@app.get("/crypto/historical")
def fetch_historical_data(
    trading_pair: str = Query("BTC/USDT"), 
    interval: str = Query("1h"), 
    data_limit: int = Query(100)
):
    client = initialize_binance_client()
    historical_data = client.fetch_ohlcv(trading_pair, interval, limit=data_limit)
    ohlcv_df = pd.DataFrame(historical_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return JSONResponse(content={"trading_pair": trading_pair, "data": ohlcv_df.to_dict(orient="records")})

@app.post("/crypto/predict")
def predict_future_price(ohlcv_data: list):
    price_data = pd.DataFrame(ohlcv_data)
    price_data["scaled_close"] = (price_data["close"] - price_data["close"].mean()) / price_data["close"].std()

    X_train, y_train = [], []
    window_size = 10
    for i in range(len(price_data) - window_size):
        X_train.append(price_data["scaled_close"].iloc[i:i + window_size].values)
        y_train.append(price_data["scaled_close"].iloc[i + window_size])
    X_train, y_train = np.array(X_train), np.array(y_train)

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    prediction = lstm_model.predict(X_train[-1].reshape(1, window_size, 1))
    return JSONResponse(content={"predicted_price": float(prediction[0][0])})

@app.post("/crypto/backtest")
def perform_backtesting(ohlcv_data: list):
    historical_df = pd.DataFrame(ohlcv_data)
    predicted_prices = []

    for idx in range(10, len(historical_df)):
        recent_segment = historical_df["close"].iloc[idx - 10:idx].values
        segment_mean, segment_std = recent_segment.mean(), recent_segment.std()
        scaled_values = (recent_segment - segment_mean) / segment_std
        simulated_prediction = np.mean(scaled_values)  
        predicted_prices.append(simulated_prediction * segment_std + segment_mean)
    
    historical_df["predicted_close"] = [None] * 10 + predicted_prices
    return JSONResponse(content=historical_df.to_dict(orient="records"))
