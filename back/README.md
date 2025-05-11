# Backend Système de Scoring de Crédit

Ce dossier contient le code du backend pour le système de scoring de crédit. Il fournit une API REST pour prédire le risque de défaut de paiement d'un client et générer des explications sur les facteurs qui influencent cette prédiction.

## Structure des fichiers

- `api.py`: Définition de l'API Flask avec les endpoints
- `prediction.py`: Logique de prédiction et calcul des valeurs SHAP
- `save_best_model.py`: Script pour entraîner et sauvegarder le meilleur modèle
- `requirements.txt`: Liste des dépendances Python nécessaires
- `data/`: Dossier pour stocker les données d'entraînement et de test
  - `application_train.csv`: Données d'entraînement (à télécharger)
  - `application_test.csv`: Données de test (à télécharger)
- `model/`: Dossier où sont sauvegardés les modèles entraînés
- `models/`: Dossier alternatif pour les modèles (pour compatibilité)

## Configuration de l'environnement

1. Créez un environnement virtuel Python :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez les jeux de données depuis [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) et placez-les dans le dossier `data/`.

## Entraînement du modèle

Pour entraîner et sauvegarder le meilleur modèle :

```bash
python save_best_model.py
```

Ce script va :
- Charger les données d'entraînement et de test
- Prétraiter les données
- Entraîner deux modèles : Random Forest et XGBoost
- Évaluer les modèles et sélectionner le meilleur
- Sauvegarder le modèle dans le dossier `model/`

## Démarrer l'API

Pour démarrer le serveur API :

```bash
python api.py
```

L'API sera accessible à l'adresse : http://localhost:5000

En production, vous pouvez utiliser Gunicorn :

```bash
gunicorn --bind 0.0.0.0:5000 api:app
```

## Endpoints de l'API

- `GET /` : Page d'accueil avec informations sur l'API
- `GET /health` : Vérification de l'état de l'API
- `GET /predict/<client_id>` : Obtenir une prédiction et des explications pour un client

## Exemple d'utilisation de l'API

Pour obtenir une prédiction pour un client (remplacez `<client_id>` par un identifiant existant) :

```bash
curl -X GET http://localhost:5000/predict/<client_id>
```

La réponse sera au format JSON avec :
- La prédiction (0 = pas de défaut, 1 = défaut)
- Les probabilités
- Les valeurs SHAP pour expliquer la prédiction

## Notes

- Les chemins des fichiers sont relatifs au dossier où l'API est lancée
- Assurez-vous que les dossiers `data/`, `model/` et `models/` existent avant de lancer les scripts 