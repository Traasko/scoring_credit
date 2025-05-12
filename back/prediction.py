import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import warnings
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join('model', 'best_model.pkl')
# Chemin vers les données clients
DATA_PATH = os.path.join('data', 'application_test.csv')

# Liste des caractéristiques utilisées pour l'entraînement du modèle
SELECTED_FEATURES = ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 
                     'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                     'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']

# Description des caractéristiques pour une meilleure interprétation
FEATURE_DESCRIPTIONS = {
    'AMT_CREDIT': 'Montant du crédit',
    'AMT_ANNUITY': 'Montant de l\'annuité',
    'DAYS_BIRTH': 'Âge (en jours)',
    'DAYS_EMPLOYED': 'Durée d\'emploi (en jours)',
    'EXT_SOURCE_1': 'Score externe 1',
    'EXT_SOURCE_2': 'Score externe 2',
    'EXT_SOURCE_3': 'Score externe 3',
    'NAME_EDUCATION_TYPE': 'Niveau d\'éducation',
    'OCCUPATION_TYPE': 'Type d\'emploi'
}

# Recommandations
FEATURE_RECOMMENDATIONS = {
    'AMT_CREDIT': 'Demander un montant de crédit moins élevé.',
    'AMT_ANNUITY': 'Opter pour une durée de remboursement plus longue pour réduire les annuités.',
    'DAYS_BIRTH': 'Ce facteur ne peut pas être modifié (âge).',
    'DAYS_EMPLOYED': 'Une stabilité professionnelle plus longue est généralement favorable.',
    'EXT_SOURCE_1': 'Améliorer votre historique de crédit auprès des organismes externes.',
    'EXT_SOURCE_2': 'Améliorer votre historique de crédit auprès des organismes externes.',
    'EXT_SOURCE_3': 'Améliorer votre historique de crédit auprès des organismes externes.',
    'NAME_EDUCATION_TYPE': 'Un niveau d\'éducation plus élevé peut être favorable.',
    'OCCUPATION_TYPE': 'Certains types d\'emploi sont considérés plus stables que d\'autres.'
}

def load_model():
    """
    Charge le modèle entraîné depuis le fichier pickle
    
    Returns:
        Model: Le modèle chargé
    """
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model
        else:
            raise FileNotFoundError(f"Le modèle n'a pas été trouvé aux chemins {MODEL_PATH}")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")

def load_data():
    """
    Charge les données de test
    
    Returns:
        DataFrame: Les données de test
    """
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except FileNotFoundError:
        raise Exception(f"Les données n'ont pas été trouvées au chemin {DATA_PATH}")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement des données: {str(e)}")

def get_client_data(client_id):
    """
    Récupère les données d'un client spécifique
    
    Args:
        client_id (int): L'identifiant du client (SK_ID_CURR)
        
    Returns:
        DataFrame: Les données du client
    """
    data = load_data()
    client_data = data[data['SK_ID_CURR'] == client_id]
    
    if client_data.empty:
        raise Exception(f"Client avec ID {client_id} non trouvé dans les données")
    
    return client_data

def preprocess_client_data(client_data):
    """
    Prétraite les données du client pour la prédiction
    
    Args:
        client_data (DataFrame): Les données du client
        
    Returns:
        DataFrame: Les données prétraitées
    """
    # Supprimer la colonne ID du client et la cible si elle existe
    features = client_data.drop(['SK_ID_CURR'], axis=1, errors='ignore')
    
    # Gérer les valeurs manquantes par type de colonne avant encodage
    for col in features.columns:
        # Vérifier si la colonne contient des valeurs manquantes
        if features[col].isnull().any():
            if features[col].dtype == 'object':
                # Pour les colonnes catégorielles, utiliser la valeur la plus fréquente (mode)
                features[col] = features[col].fillna(features[col].mode()[0] if not features[col].mode().empty else "MISSING")
            else:
                # Pour les colonnes numériques, utiliser la médiane
                features[col] = features[col].fillna(features[col].median() if not pd.isna(features[col].median()) else 0)
    
    # Encoder les variables catégorielles
    le = LabelEncoder()
    for col in features.columns:
        if features[col].dtype == 'object':
            # Convertir en chaîne pour éviter les erreurs avec les types mixtes
            features[col] = features[col].astype(str)
            features[col] = le.fit_transform(features[col])
    
    # Vérification finale pour s'assurer qu'il ne reste pas de NaN
    # Remplacer tous les NaN restants par 0
    features = features.fillna(0)
    
    # Vérifier qu'il ne reste plus de NaN
    assert not features.isnull().any().any(), "Il reste des valeurs NaN après prétraitement"
    
    # IMPORTANT: Sélectionner uniquement les caractéristiques utilisées pour l'entraînement
    # Vérifier si toutes les caractéristiques requises sont présentes
    missing_features = [f for f in SELECTED_FEATURES if f not in features.columns]
    if missing_features:
        print(f"Attention: Les caractéristiques suivantes sont manquantes: {missing_features}")
        for feature in missing_features:
            features[feature] = 0  # Ajouter des valeurs par défaut
    
    # Ne garder que les caractéristiques utilisées pour l'entraînement
    features = features[SELECTED_FEATURES]
    
    return features

def compute_shap_values(model, features):
    """
    Calcule les valeurs SHAP pour expliquer les prédictions
    
    Args:
        model: Le modèle utilisé pour la prédiction
        features (DataFrame): Les caractéristiques du client
    
    Returns:
        dict: Dictionnaire contenant les valeurs SHAP et leurs informations
    """
    try:
        # Créer un explainer SHAP approprié au modèle
        if hasattr(model, 'predict_proba'):
            if str(type(model)).find('XGBoost') >= 0:
                explainer = shap.Explainer(model)
            else:
                # Pour RandomForest et autres modèles
                explainer = shap.TreeExplainer(model)
            
            # Calculer les valeurs SHAP
            shap_values = explainer.shap_values(features)
            
            # Si shap_values est une liste (pour les cas à plusieurs classes), prendre la classe 1 (défaut de paiement)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Valeurs pour la classe 1 (défaut de paiement)
            
            # Convertir en dictionnaire plus facile à utiliser
            shap_dict = {}
            feature_importance = []
            
            for i, feature in enumerate(SELECTED_FEATURES):
                feature_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                shap_value = float(shap_values[0][i])  # Convertir en float pour être serializable en JSON
                recommendation = FEATURE_RECOMMENDATIONS.get(feature, '')
                
                # Déterminer si cette caractéristique a un impact positif ou négatif
                impact = "positif" if shap_value < 0 else "négatif"  # Inversé car une valeur négative signifie une probabilité plus faible de défaut
                
                feature_info = {
                    'feature': feature,
                    'feature_name': feature_name,
                    'value': float(features[feature].iloc[0]),
                    'shap_value': shap_value,
                    'impact': impact,
                    'recommendation': recommendation
                }
                feature_importance.append(feature_info)
            
            # Trier par amplitude de l'impact (valeur absolue)
            feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            shap_dict['feature_importance'] = feature_importance
            
            # Générer un graphique des valeurs SHAP
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            
            # Sauvegarder le graphique en base64 pour l'afficher dans le frontend
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()
            
            shap_dict['shap_plot'] = img_base64
            
            return shap_dict
        else:
            return {"error": "Le modèle ne prend pas en charge les explications SHAP"}
    except Exception as e:
        print(f"Erreur lors du calcul des valeurs SHAP: {str(e)}")
        return {"error": str(e)}

def make_prediction(client_id):
    """
    Effectue une prédiction pour un client spécifique
    
    Args:
        client_id (int): L'identifiant du client (SK_ID_CURR)
        
    Returns:
        tuple: (prédiction, probabilités, explications SHAP)
    """
    # Charger le modèle
    model = load_model()
    
    try:
        # Obtenir les données du client
        client_data = get_client_data(client_id)
        
        # Prétraiter les données
        features = preprocess_client_data(client_data)
        
        # Debug: Afficher le nombre de caractéristiques
        print(f"Nombre de caractéristiques utilisées pour la prédiction: {features.shape[1]}")
        print(f"Caractéristiques: {list(features.columns)}")
        
        # Faire la prédiction
        prediction = model.predict(features)
        
        # Obtenir les probabilités
        probabilities = model.predict_proba(features)
        
        # Calculer les valeurs SHAP pour l'explication
        shap_explanation = compute_shap_values(model, features)
        
        return prediction[0], probabilities[0], shap_explanation
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        raise e

def preprocess_new_data(new_data):
    """
    Prétraite les nouvelles données soumises via l'API
    """
    # Vérifier que toutes les caractéristiques requises sont présentes
    missing_features = [f for f in SELECTED_FEATURES if f not in new_data.columns]
    if missing_features:
        raise ValueError(f"Caractéristiques manquantes: {missing_features}")
    
    # Sélectionner uniquement les caractéristiques nécessaires
    features = new_data[SELECTED_FEATURES].copy()
    
    # Gérer les valeurs manquantes
    for col in features.columns:
        if features[col].isnull().any():
            if features[col].dtype == 'object':
                features[col] = features[col].fillna("MISSING")
            else:
                features[col] = features[col].fillna(0)
    
    # Encoder les variables catégorielles
    le = LabelEncoder()
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = features[col].astype(str)
            features[col] = le.fit_transform(features[col])
    
    # Vérification finale
    features = features.fillna(0)
    
    return features

if __name__ == "__main__":
    # Test de la fonction de prédiction
    try:
        client_id = 100001  # ID de test
        prediction, probabilities, shap_values = make_prediction(client_id)
        print(f"Prédiction pour le client {client_id}: {prediction}")
        print(f"Probabilités: {probabilities}")
        print(f"Top 3 facteurs influençant la prédiction:")
        for i, factor in enumerate(shap_values['feature_importance'][:3]):
            print(f"  {i+1}. {factor['feature_name']}: Impact {factor['impact']} - {factor['recommendation']}")
    except Exception as e:
        print(f"Erreur: {str(e)}") 