from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os

# Load the LSTM model
model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")  # Make sure this file is pushed to GitHub or added via Render's disk

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allows external requests (like from Flutter or Postman)

@app.route("/", methods=["GET"])
def home():
    return "Stock Prediction API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json['input']
        input_data = np.array(data).reshape(1, 60, 7)  # Adjust shape
        prediction = model.predict(input_data).tolist()
        return jsonify({
            'prediction': prediction,
            'reason': 'Predicted using 60 days of technical indicators with LSTM model'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app (important: use dynamic port for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the port assigned by Render
    app.run(host="0.0.0.0", port=port)

