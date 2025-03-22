from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

app = Flask(__name__)
CORS(app)

# Load pre-trained LSTM model
model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")

def fetch_data_with_indicators(ticker):
    df = yf.download(ticker, period='90d', interval='1d', group_by='column')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if 'Close' not in df.columns or len(df) < 60:
        raise ValueError("Not enough data or invalid symbol.")

    df = df.tail(60).copy()

    # Calculate technical indicators
    df['SMA_10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    bb = BollingerBands(df['Close'], window=20)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    df = df.dropna().tail(60)

    # Final selected features
    selected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
                        'SMA_10', 'SMA_20', 'BB_upper', 'BB_lower']

    return df[selected_columns]

@app.route("/", methods=["GET"])
def home():
    return "Stock LSTM Prediction API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.json.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400

        print(f"Fetching data for: {symbol}")
        df = fetch_data_with_indicators(symbol)

        if df.shape[0] < 60:
            return jsonify({"error": "Not enough data to make a prediction"}), 400

        # Fill any remaining NaNs
        df.fillna(0, inplace=True)

        # Prepare input for model
        input_data = df.values  # shape (60, features)
        input_reshaped = np.expand_dims(input_data, axis=0)  # shape (1, 60, features)

        print("Running prediction...")
        prediction = model.predict(input_reshaped)
        predicted_price = float(prediction[0][0])

        return jsonify({
            "symbol": symbol.upper(),
            "prediction": [[predicted_price]],
            "reason": f"Predicted using LSTM model with technical indicators for {symbol}"
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
