from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model, scaler, and columns
model = joblib.load('./models/model-rf.pkl')
scaler = joblib.load('./models/scaler-rf.pkl')
trained_columns = joblib.load('./models/trained_columns.pkl')

def preprocess_input(data, trained_columns):
    """
    Preprocess the input data for the model. This includes ensuring that the input
    columns match the columns used during model training.
    """
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])
    
    # One-hot encode input data to match trained columns
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Add missing columns from training
    for col in trained_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure input_df has the same columns and order as during training
    input_df = input_df[trained_columns]
    
    return input_df

@app.route('/')
def home():
    return "Airbnb Price Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions using the trained model.
    """
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Preprocess the input data to match the trained model's format
        processed_input = preprocess_input(data, trained_columns)
        
        # Scale the input data
        scaled_input = scaler.transform(processed_input)
        
        # Make a prediction
        prediction = model.predict(scaled_input)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
