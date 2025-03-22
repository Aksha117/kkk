from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")

@app.route("/")
def home():
    return "LSTM Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.json.get("symbol")
        if not symbol:
            return jsonify({"error": "Stock symbol is required."}), 400

        # Get stock data
        df = yf.download(symbol, period="90d", interval="1d")
        if df.empty or len(df) < 60:
            return jsonify({"error": "Invalid symbol or insufficient data."}), 400

        df = df.tail(60).copy()

        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(0)

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD'] = df['MACD'].fillna(0)

        # Prepare features
        features = np.column_stack([
            df['Open'].values,
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            df['Volume'].values,
            df['RSI'].values,
            df['MACD'].values
        ])

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        input_data = scaled_features.reshape(1, 60, 7)

        prediction = model.predict(input_data)
        scaled_prediction = prediction[0][0]
        inverse_scaled = scaler.inverse_transform([[0, 0, 0, scaled_prediction, 0, 0, 0]])[0][3]

        return jsonify({
            "symbol": symbol,
            "predicted_close_price": round(inverse_scaled, 2)
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
