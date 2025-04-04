from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the models
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('rf_expenditure_model.pkl', 'rb') as file:
    rf_expenditure_model = pickle.load(file)

with open('rf_fiscal_deficit_model (1).pkl', 'rb') as file:
    rf_fiscal_deficit_model = pickle.load(file)

with open('xgb_expenditure_model.pkl', 'rb') as file:
    xgb_expenditure_model = pickle.load(file)

with open('xgb_fiscal_deficit_model.pkl', 'rb') as file:
    xgb_fiscal_deficit_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['features']])
    
    # Scale the input features
    scaled_features = scaler.transform(features)

    # Extract total revenue from input
    total_revenue = data['features'][0]

    # Model predictions
    rf_exp = rf_expenditure_model.predict(scaled_features)[0]
    xgb_exp = xgb_expenditure_model.predict(scaled_features)[0]
    ensemble_exp = (rf_exp + xgb_exp) / 2

    # Fiscal Deficit Predictions
    rf_deficit = rf_fiscal_deficit_model.predict(scaled_features)[0]
    xgb_deficit = xgb_fiscal_deficit_model.predict(scaled_features)[0]
    ensemble_deficit = (rf_deficit + xgb_deficit) / 2

    return jsonify({
        "Random Forest Expenditure": f"{rf_exp:,.2f}",
        "XGBoost Expenditure": f"{xgb_exp:,.2f}",
        "Ensemble Expenditure": f"{ensemble_exp:,.2f}",
        "Random Forest Fiscal Deficit": f"{rf_deficit:,.2f}",
        "XGBoost Fiscal Deficit": f"{xgb_deficit:,.2f}",
        "Ensemble Fiscal Deficit": f"{ensemble_deficit:,.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)

