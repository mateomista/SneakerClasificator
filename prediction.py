from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
import numpy as np

model_path = './Model/predictSneakersModelPreTrainned.h5'

def load_trained_model(model_path):

    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def preprocess_image(image, target_size=(224, 224)):
    
    img = image.resize(target_size)
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)  
    
    return np.expand_dims(img_array, axis=0)

def predict(model, image):
 
    if model is None:
        print("Model is not loaded. Cannot make predictions.")
        return None
    
    try:
        image = preprocess_image(image)
        predictions = model.predict(image)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
def get_prediction_label(predictions):

    if predictions is None or len(predictions) == 0:
        print("No predictions to process.")
        return None
    
    predicted_label = np.argmax(predictions, axis=1)[0]
    return predicted_label

def get_prediction_confidence(predictions):
   
    if predictions is None or len(predictions) == 0:
        print("No predictions to process.")
        return None
    
    confidence_score = np.max(predictions, axis=1)[0]
    return confidence_score

predicted_model = load_trained_model(model_path)
if predicted_model is None:
    print("Failed to load the model. Exiting.")

image = Image.open('./sneakers-dataset/sneakers-dataset/adidas_samba/0005.jpg');
predictions = predict(predicted_model, image)
predicted_label = get_prediction_label(predictions)
predicted_confidence = get_prediction_confidence(predictions)
print(f"Predicted Label: {predicted_label}")
print(f"Confidence Score: {predicted_confidence:.2f}")