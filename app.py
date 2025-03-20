from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

# Load the LSTM model
model = tf.keras.models.load_model("nse_lstm_model_fixed.h5")  # Ensure this file is uploaded

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allows external requests

@app.route("/", methods=["GET"])
def home():
    return "Stock Prediction API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json['input']
        input_data = np.array(data).reshape(1, 60, 7)  # Adjust shape
        prediction = model.predict(input_data).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
