from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__, static_folder='static')

# Charger le modèle de détection de tumeurs
model = load_model(r'C:\Users\KOURCHTE Oumaima\Desktop\PFE\brain_tumor_model.keras')

# Définir le chemin du dossier temporaire
temp_dir = 'temp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Fonction de prétraitement d'image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-tumor', methods=['POST'])
def detect_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    img_path = os.path.join(temp_dir, 'temp_image.jpg')  # Chemin temporaire pour enregistrer l'image
    file.save(img_path)

    # Prétraiter l'image
    preprocessed_image = preprocess_image(img_path)

    # Faire la prédiction
    prediction = model.predict(preprocessed_image)

    # Interpréter la prédiction (par exemple, 1 pour tumeur, 0 pour pas de tumeur)
    result = 'Tumor Detected' if prediction > 0.5 else 'No Tumor Detected'

    return jsonify({'result': result})

# Démarrer le serveur Flask si le fichier est exécuté directement
if __name__ == '__main__':
    app.run(debug=True)
