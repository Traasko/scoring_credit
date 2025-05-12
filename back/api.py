from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from prediction import make_prediction, preprocess_new_data, compute_shap_values, load_model

# Initialisation de l' API Flask
app = Flask(__name__)
CORS(app)

model_path = os.path.join('model', 'best_model.pkl')

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict(client_id):
    """
    Endpoint pour obtenir la prédiction pour un client spécifique.
    
    Args:
        client_id (int): Identifiant du client (SK_ID_CURR)
        
    Returns:
        JSON: Résultat de la prédiction avec les probabilités et les explications SHAP
    """
    try:
        prediction, probabilities, shap_values = make_prediction(client_id)
        
        response = {
            'sk_id': client_id,
            'prediction': int(prediction),
            'proba': probabilities.tolist(),
            'shap_values': shap_values
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    """
    Endpoint racine avec des instructions d'utilisation
    """
    return jsonify({
        'name': 'Système de Scoring de Crédit API',
        'version': '1.0.0',
        'endpoints': [
            {
                'path': '/predict/<client_id>',
                'method': 'GET',
                'description': 'Obtenir une prédiction pour un client'
            }
        ]
    })

@app.route('/predict_new', methods=['POST'])
def predict_new():
    """
    Endpoint pour prédire le risque d'une nouvelle demande de crédit
    """
    try:
        data = request.get_json()
        
        new_data = pd.DataFrame([data])
        
        features = preprocess_new_data(new_data)
        
        model = load_model()
        
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        shap_explanation = compute_shap_values(model, features)
        
        response = {
            'prediction': int(prediction[0]),
            'proba': probabilities[0].tolist(),
            'shap_values': shap_explanation
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Vérifier si le modèle existe
    model_exists = os.path.exists(model_path)
    if not model_exists:
        print(f"ATTENTION: Le modèle n'existe pas aux chemins {model_path}")
        print("Assurez-vous d'avoir entraîné le modèle avant de démarrer l'API")
        print("Exécutez d'abord: python save_best_model.py")
    else:
        if os.path.exists(model_path):
            print(f"Modèle trouvé: {model_path}")
    
    # Démarrer l'application sur le port 5000
    app.run(host='0.0.0.0', port=5000, debug=True) 