from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD

app = Flask(__name__)
CORS(app)

# Load your trained LSTM model
model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")

def fetch_data_with_indicators(ticker):
    df = yf.download(ticker, period='90d', interval='1d', group_by='column')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if df.empty or 'Close' not in df.columns:
        raise ValueError("Invalid symbol or no data found.")

    df = df.tail(60).copy()

    # Calculate 7 indicators
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()

    # Fill missing values
    df.fillna(0, inplace=True)

    # Ensure only the 7 required columns
    selected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
    return df[selected_columns].tail(60)

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

        if df.shape != (60, 7):
            return jsonify({"error": "Data shape mismatch or not enough data"}), 400

        input_data = np.expand_dims(df.values, axis=0)  # shape (1, 60, 7)

        print("Running prediction...")
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0][0])

        return jsonify({
            "symbol": symbol.upper(),
            "prediction": [[predicted_price]],
            "reason": "Predicted using 60-day historical data and technical indicators."
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
