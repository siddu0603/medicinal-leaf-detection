import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("leaves_detect.h5")

# Load class labels
class_indices = {0: 'Alpinia Galanga (Rasna)', 1: 'Amaranthus Viridis (Arive-Dantu)', 
 2: 'Artocarpus Heterophyllus (Jackfruit)', 3: 'Azadirachta Indica (Neem)', 
 4: 'Basella Alba (Basale)', 5: 'Brassica Juncea (Indian Mustard)', 
 6: 'Carissa Carandas (Karanda)', 7: 'Citrus Limon (Lemon)', 
 8: 'Ficus Auriculata (Roxburgh fig)', 9: 'Ficus Religiosa (Peepal Tree)', 
 10: 'Hibiscus Rosa-sinensis', 11: 'Jasminum (Jasmine)', 
 12: 'Mangifera Indica (Mango)', 13: 'Mentha (Mint)', 
 14: 'Moringa Oleifera (Drumstick)', 15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)', 
 16: 'Murraya Koenigii (Curry)', 17: 'Nerium Oleander (Oleander)', 
 18: 'Nyctanthes Arbor-tristis (Parijata)', 19: 'Ocimum Tenuiflorum (Tulsi)', 
 20: 'Piper Betle (Betel)', 21: 'Plectranthus Amboinicus (Mexican Mint)', 
 22: 'Pongamia Pinnata (Indian Beech)', 23: 'Psidium Guajava (Guava)', 
 24: 'Punica Granatum (Pomegranate)', 25: 'Santalum Album (Sandalwood)', 
 26: 'Syzygium Cumini (Jamun)', 27: 'Syzygium Jambos (Rose Apple)', 
 28: 'Tabernaemontana Divaricata (Crape Jasmine)', 29: 'Trigonella Foenum-graecum (Fenugreek)'}

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Change to (224, 224) if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize (MobileNetV2 expects [-1, 1])
    img_array /= 255.0  # Remove preprocess_input if training was done in [0,1] range
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')    

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Predict
    prediction = model.predict(img_array)

    # Get top prediction
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)  # Get confidence score
    class_name = class_indices.get(class_idx, "Unknown")

    # Debugging info
    print(f"Predicted Index: {class_idx}, Class: {class_name}, Confidence: {confidence:.2f}")

    return jsonify({'prediction': class_name, 'confidence': float(confidence)})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
