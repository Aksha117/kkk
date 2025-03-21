from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the pre-trained LSTM model
model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")

# 📊 RSI Calculation
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 📉 MACD Calculation
def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    return macd

# 📈 Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band

@app.route("/", methods=["GET"])
def home():
    return "Stock LSTM Prediction API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.json.get('symbol')
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400

        # 🧾 Fetch historical data from Yahoo Finance
        stock = yf.Ticker(f"{symbol}.NS")
        df = stock.history(period="3mo")

        if len(df) < 60:
            return jsonify({"error": "Not enough data for prediction"}), 400

        # 🧠 Compute technical indicators
        df['RSI'] = calculate_rsi(df)
        df['MACD'] = calculate_macd(df)
        df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)

        # Keep last 60 rows
        df = df.tail(60)

        # Replace NaN values with 0
        df.fillna(0, inplace=True)

        # 🔢 Prepare input: [open, high, low, close, volume, RSI, MACD]
        input_data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].values
        input_reshaped = np.expand_dims(input_data, axis=0)  # Shape: [1, 60, 7]

        # 🤖 Predict
        prediction = model.predict(input_reshaped)
        predicted_price = float(prediction[0][0])

        return jsonify({
            "symbol": symbol,
            "prediction": [[predicted_price]],
            "reason": f"Predicted using 60-day technical indicators with LSTM model for {symbol}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
