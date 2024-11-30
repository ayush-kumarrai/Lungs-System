import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('lung_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def preprocess_input(data):
    """
    Robust input preprocessing with detailed validation
    """
    # Mapping for encoding
    encodings = {
        'gender': {'M': 1, 'F': 0},
        'binary_features': {'YES': 2, 'NO': 1}
    }
    
    # Validate and convert inputs
    processed_data = []
    
    # Gender conversion
    processed_data.append(encodings['gender'].get(data['gender'], 0))
    
    # Age validation
    try:
        age = int(data['age'])
        processed_data.append(max(0, min(age, 120)))  # Clamp age between 0-120
    except (ValueError, TypeError):
        processed_data.append(40)  # Default age if invalid
    
    # Binary features conversion
    binary_features = [
        'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', 
        'chronic_disease', 'fatigue', 'allergy', 'wheezing', 
        'alcohol', 'coughing', 'shortness_of_breath', 
        'swallowing_difficulty', 'chest_pain'
    ]
    
    for feature in binary_features:
        processed_data.append(
            encodings['binary_features'].get(data.get(feature, 'NO'), 1)
        )
    
    return np.array(processed_data).reshape(1, -1)

@app.route('/api/predict', methods=['POST'])
def predict_lung_cancer():
    try:
        data = request.json
        
        # Preprocess input
        input_array = preprocess_input(data)
        
        # Predict
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)
        
        return jsonify({
            'prediction': 'Positive' if prediction[0] == 1 else 'Negative',
            'probability': float(probability[0][prediction[0]])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)