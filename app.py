from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import ccxt
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os


app = FastAPI(title="Crypto Price Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")


def get_binance_client():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET
    })


@app.get("/crypto/historical")
def get_historical_data(
    symbol: str = Query("BTC/USDT"), 
    timeframe: str = Query("1h"), 
    limit: int = Query(100)
):
    client = get_binance_client()
    data = client.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

    return JSONResponse(content={"symbol": symbol, "data": df.to_dict(orient="records")})


@app.post("/crypto/predict")
def predict_price(data: list):
    df = pd.DataFrame(data)
    df["scaled_close"] = (df["close"] - df["close"].mean()) / df["close"].std()


    X, y = [], []
    lookback = 10
    for i in range(len(df) - lookback):
        X.append(df["scaled_close"].iloc[i:i + lookback].values)
        y.append(df["scaled_close"].iloc[i + lookback])
    X, y = np.array(X), np.array(y)


    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    prediction = model.predict(X[-1].reshape(1, lookback, 1))
    return JSONResponse(content={"prediction": float(prediction[0][0])})


@app.post("/crypto/backtest")
def backtest(data: list):
    df = pd.DataFrame(data)
    predictions = []
    for i in range(10, len(df)):
        segment = df["close"].iloc[i - 10:i].values
        mean, std = segment.mean(), segment.std()
        scaled_segment = (segment - mean) / std
        prediction = np.mean(scaled_segment)  
        predictions.append(prediction * std + mean)
    
    df["predicted_close"] = [None] * 10 + predictions
    return JSONResponse(content=df.to_dict(orient="records"))
