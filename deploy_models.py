from flask import Flask, request, jsonify
import lightgbm as lgb
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
lightgbm_model = lgb.Booster(model_file="lightgbm_model.txt")
xgboost_model = xgb.Booster()
xgboost_model.load_model("xgboost_model.json")
with open("random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

# Utility function to preprocess input
def preprocess_input(data):
    """
    Preprocess input data to match the format required by the models.
    Assumes data is sent as JSON with column names matching the training set.
    """
    df = pd.DataFrame(data)
    return df

# LightGBM prediction endpoint
@app.route("/predict/lightgbm", methods=["POST"])
def predict_lightgbm():
    try:
        input_data = request.json
        input_df = preprocess_input(input_data)
        predictions = lightgbm_model.predict(input_df)
        predictions = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# XGBoost prediction endpoint
@app.route("/predict/xgboost", methods=["POST"])
def predict_xgboost():
    try:
        input_data = request.json
        input_df = preprocess_input(input_data)
        dmatrix = xgb.DMatrix(input_df)
        predictions = xgboost_model.predict(dmatrix)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Random Forest prediction endpoint
@app.route("/predict/randomforest", methods=["POST"])
def predict_randomforest():
    try:
        input_data = request.json
        input_df = preprocess_input(input_data)
        predictions = random_forest_model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "All models are ready for predictions!"})

# Main function to run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)