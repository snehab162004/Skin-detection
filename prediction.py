import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("skin_disease_model.h5")

# Define a function to preprocess the image and make predictions
def predict_image(img_path, model, img_size=128):
    # Load the image
    img = image.load_img(img_path, target_size=(img_size, img_size))

    # Convert the image to a numpy array and normalize it
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict the class of the image
    predictions = model.predict(img_array)

    # Get the class names from the directory
    class_names = os.listdir('./processed_dataset/train')  # Ensure your dataset structure matches
    if len(predictions[0]) != len(class_names):
        raise ValueError(f"Model output size ({len(predictions[0])}) does not match number of classes ({len(class_names)}).")

    # Get the predicted class index and confidence score
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence_score = predictions[0][predicted_index] * 100  # Convert to percentage

    return predicted_class, confidence_score, predictions

# Function to handle file upload and prediction
def upload_and_predict(img_path):
    if os.path.exists(img_path):  # Check if the file exists
        predicted_class, confidence_score, predictions = predict_image(img_path, model)

        # Display the prediction results
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence Score: {confidence_score:.2f}%")  # Display accuracy
        print(f"Prediction Probabilities: {predictions}")

        # Show the image with the prediction
        img = image.load_img(img_path, target_size=(128, 128))
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence_score:.2f}%)")
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
    else:
        print("File not found, please upload a valid image.")

# Example usage
img_path = input("Please enter the path to your skin image: ")
upload_and_predict(img_path)
