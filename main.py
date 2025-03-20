from flask import Flask, request, jsonify
import joblib
import numpy as np
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON
        data = request.get_json()

        # Extract text input
        input_text = data.get('text', '')

        # Validate input
        if not isinstance(input_text, str):
            return jsonify({'error': 'Invalid input format. Expected a string.'}), 400

        # Ensure input is a list of strings before TF-IDF transformation
        transformed_data = vectorizer.transform([input_text])

        # Ensure transformed data is in correct format
        if isinstance(transformed_data, np.ndarray):
            transformed_data = csr_matrix(transformed_data)  # Convert to sparse matrix if needed

        # Debugging logs
        print(f"✅ Transformed Data Type: {type(transformed_data)}")
        print(f"✅ Transformed Data Shape: {transformed_data.shape}")

        # Make a prediction
        prediction = model.predict(transformed_data)[0]

        # Return prediction result
        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")  # Print error in logs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
