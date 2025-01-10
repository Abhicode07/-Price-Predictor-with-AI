

# Crypto Price Prediction API

This is a FastAPI-based backend for fetching cryptocurrency historical data, predicting future prices, and backtesting models.

## Features

- Fetch historical data from Binance.
- Predict future cryptocurrency prices using an LSTM model.
- Backtest predictions using historical data.
- CORS support for frontend integration.

---

## Installation

### Prerequisites

- Python 3.9+
- Binance API key and secret (create one(https://www.binance.com/en/my/settings/api-management)).

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo-name/crypto-price-prediction-backend.git
   cd crypto-price-prediction-backend

2. Create a virtual environment and activate it:

   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  

  pip install -r requirements.txt
  Create a .env file in the root directory with the following content:
  BINANCE_API_KEY=your_api_key
  BINANCE_API_SECRET=your_api_secret

3. Run server
    uvicorn main:app --reload



### API Endpoints
1. GET /crypto/historical: Fetch historical cryptocurrency data.
Parameters:
symbol: Trading pair (e.g., BTC/USDT).
timeframe: Timeframe for data (e.g., 1h).
limit: Number of data points


2. Predict Future Prices
POST /crypto/predict
Request Body: Historical OHLCV data as a list of objects.


3. Backtest Predictions
POST /crypto/backtest
Request Body: Historical OHLCV data as a list of objects.