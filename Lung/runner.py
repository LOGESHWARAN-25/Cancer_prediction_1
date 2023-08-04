import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict_lung": {"origins": "*"}})


model = tf.keras.models.load_model("Lung and Colon Cancer - MobileNetV3.h5")
class_names = ['Pneumonia','Normal']

def download_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Failed to download image from URL")


def load_image(image_path_or_url):
    # image_data = download_image_from_url(image_path_or_url)
    # image = Image.open(BytesIO(image_data))
    image = Image.open(image_path_or_url)
    return image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert('RGB')  # Convert to RGB mode for PNG images
    image = np.array(image)   # Normalize the pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)
    return image
@app.route('/predict_lung', methods=['POST'])
def predict_cancer_type():
    print("req in lung")
    data = request.get_json()
    image_path_or_url = data.get('image_path_or_url')
    # image_path_or_url = 'C:/Users/Loges/Downloads/archive (2)/Multi Cancer/Brain Cancer/brain_menin/brain_menin_0016.jpg'
    image = load_image(image_path_or_url)

    # Preprocess the image
    image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(image)
    print(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    # Check if the predicted class is in the list of class names from the training dataset
    if predicted_class in class_names:
        # Load the image for visualization
        # image_for_display = cv2.cvtColor(cv2.imread(image_path_or_url), cv2.COLOR_BGR2RGB)

        # Get the probability score for the predicted class
        predicted_prob = predictions[0][predicted_class_index]

        # Prepare the text to be displayed on the image
        # text = f"Predicted: {predicted_class}\nProbability: {predicted_prob:5.3f}"

        # Resize the image and create a blank canvas for the enlarged image
        # enlarged_image = cv2.resize(image_for_display, (800, 800))
        # canvas = np.ones_like(enlarged_image) * 255

        # Calculate the position for the text on the canvas
        text_position = (20, 40)

        # Paste the enlarged image onto the canvas
        # canvas[:enlarged_image.shape[0], :enlarged_image.shape[1], :] = enlarged_image

        # Display the image with the predicted class and probability
        return jsonify({'prediction': predicted_class})
        # return predicted_class
    else:
        print("Wrong Input: The provided image does not belong to any of the trained classes.")
        return "Wrong Input"

if __name__ == '__main__':
    app.run(port= 5003,debug=True)
    # print(predict_cancer_type())