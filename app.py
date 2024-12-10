from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained model
if not os.path.exists("skin_disease_model.h5"):
    raise FileNotFoundError("The model file 'skin_disease_model.h5' is missing.")
model = tf.keras.models.load_model("skin_disease_model.h5")

# Ensure the uploads directory exists
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Define a function to preprocess the image and make predictions
def predict_image(img_path, model, img_size=128):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    print(f"Predictions: {predictions}")  # Debugging
    
    # Ensure predictions match class_names
    class_names = ["acne", "eczema", "hemophilia", "normal", "psoriasis", "vitiligo"]
    if len(predictions[0]) != len(class_names):
        raise ValueError(f"Model output size ({len(predictions[0])}) does not match number of classes ({len(class_names)}).")
    
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, predictions[0]

def preprocess_and_predict(image_data, model, img_size=128):
    image_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_names = ["acne","eczema","hemophilia","normal","psoriasis","vitiligo"] 
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, predictions[0]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nextpage.html')
def nextpage():
    return render_template('nextpage.html')

@app.route('/realtime.html')
def realTime():
    return render_template('realtime.html')

@app.route('/realtime-predict', methods=['POST'])
def realtime_predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    try:
        image_data = data['image']
        predicted_class, predictions = preprocess_and_predict(image_data, model)
        return jsonify({
            'predicted_class': predicted_class,
            'predictions': predictions.tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        img_path = os.path.join(uploads_dir, file.filename)
        file.save(img_path)
        try:
            predicted_class, predictions = predict_image(img_path, model)
        finally:
            os.remove(img_path)  # Cleanup uploaded file
        return jsonify({
            'predicted_class': predicted_class,
            'predictions': predictions.tolist()
        }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
