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

# Load model
try:
    model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load error:", e)

# Fetch data and calculate indicators
def fetch_data_with_indicators(ticker):
    try:
        print(f"📥 Downloading data for: {ticker}")
        df = yf.download(ticker, period='90d', interval='1d', group_by='column')
        print("✅ Data downloaded")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        print("Columns:", df.columns.tolist())

        if df.empty or 'Close' not in df.columns:
            raise ValueError("Invalid symbol or no data found.")

        df = df.tail(60).copy()
        print("Rows after tail(60):", len(df))

        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()

        df.fillna(0, inplace=True)
        print("✅ Indicators calculated and NaNs filled")

        selected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
        return df[selected_columns].tail(60)

    except Exception as e:
        print("❌ ERROR in fetch_data_with_indicators:", str(e))
        raise

# Home route
@app.route("/", methods=["GET"])
def home():
    return "Stock LSTM Prediction API is running!", 200

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.json.get("symbol")
        print("🔎 Predict request received for:", symbol)

        df = fetch_data_with_indicators(symbol)
        print("✅ Data shape:", df.shape)

        input_data = np.expand_dims(df.values, axis=0)
        print("Input shape for model:", input_data.shape)

        prediction = model.predict(input_data)
        predicted_price = float(prediction[0][0])
        print("✅ Prediction:", predicted_price)

        return jsonify({
            "symbol": symbol.upper(),
            "prediction": [[predicted_price]],
            "reason": "Predicted using 60-day technical indicators and LSTM"
        })

    except Exception as e:
        print("❌ ERROR in /predict route:", str(e))
        return jsonify({"error": str(e)}), 500

# Start app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
