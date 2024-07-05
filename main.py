# Dependencies
import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import matplotlib.pyplot as plt

class_labels = {
    'batteries': 0,
    'clothes': 1,
    'e-waste': 2,
    'glass': 3,
    'light blubs': 4,
    'metal': 5,
    'organic': 6,
    'paper': 7,
    'plastic': 8
}

def getPrediction(filename):
    newmodel = tf.keras.models.load_model("Resources/Model/final/vgg16_final.h5")  # Replace with the actual path to your model
    #newmodel = tf.keras.models.load_model("Resources/Model/final/inception_final.h5")

    if isinstance(filename, str):
        # Load and preprocess the image if the input is a file path
        img = load_img('static/' + filename, target_size=(150, 150))  # Adjust target_size as needed vgg 
        #img = load_img('static/' + filename, target_size=(220, 220))   # Adjust target_size as needed inception
        img_array = img_to_array(img)
        
    elif isinstance(filename, Image.Image):
        # If the input is already an image object, convert it to an array
        img_array = img_to_array(filename)
        
    else:
        raise ValueError("Invalid input type. Provide either a file path or a PIL Image object.")

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = newmodel.predict(img_array)

    # Interpret the predictions
    predicted_class = np.argmax(predictions, axis=1)[0]
    probability = predictions[0][predicted_class]

    # Get the class label
    predicted_label = [key for key, value in class_labels.items() if value == predicted_class][0]

    predicted_label = str(predicted_label)
    probability = str(probability)

    values = [predicted_label, probability, filename]
    return values[0], values[1], values[2]
