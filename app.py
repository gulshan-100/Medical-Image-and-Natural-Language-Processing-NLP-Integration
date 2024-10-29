from flask import Flask, request, render_template, jsonify, current_app
import os
import numpy as np
import logging
import traceback

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Create uploads directory
os.makedirs('uploads', exist_ok=True)

# Define class labels
class_labels = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor', 
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

# Load model with error handling
try:
    model = load_model('tumor_classification.h5', compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"})

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type"})
    
    try:
        # Save and process image
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        result = class_labels[predicted_class]

        # Clean up
        os.remove(img_path)

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        if 'img_path' in locals() and os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": str(e)})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Print raw request data for debugging
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Request data: {request.get_data()}")
        
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        logger.debug(f"Parsed JSON data: {data}")

        if not data or 'medical_text' not in data:
            logger.error("No medical text in request")
            return jsonify({"error": "No medical text provided"}), 400

        medical_text = data['medical_text']
        if not medical_text.strip():
            logger.error("Empty medical text")
            return jsonify({"error": "Medical text cannot be empty"}), 400

        # Import and generate report
        from nlp_utils import generate_medical_report
        logger.debug("Calling generate_medical_report function")
        
        report = generate_medical_report(medical_text)
        logger.debug(f"Generated report: {report}")
        
        if isinstance(report, dict) and "error" in report:
            logger.error(f"Error in report generation: {report['error']}")
            return jsonify({"error": report['error']}), 500
            
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Exception in generate_report: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/test_report', methods=['GET'])
def test_report():
    try:
        from nlp_utils import generate_medical_report
        test_text = "Patient has fever and headache. Taking aspirin."
        report = generate_medical_report(test_text)
        return jsonify({"status": "success", "report": report})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
