import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Définir les chemins de fichiers
DATA_DIR = 'data'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Chemins des fichiers d'entraînement et de test
TRAIN_PATH = os.path.join(DATA_DIR, 'application_train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'application_test.csv')

# Liste des caractéristiques à utiliser (réduites pour la simplicité et la performance)
SELECTED_FEATURES = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'DAYS_BIRTH', 
                    'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                    'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']

def load_and_preprocess_data():
    """
    Charge et prétraite les données d'entraînement et de test
    """
    print("Chargement des données...")
    # Vérifier si les fichiers existent
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Fichier d'entraînement non trouvé: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Fichier de test non trouvé: {TEST_PATH}")

    # Charger les données
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Données chargées: {train_df.shape[0]} lignes d'entraînement, {test_df.shape[0]} lignes de test")
    
    # Extraire les variables cibles et les identifiants
    y = train_df['TARGET']
    train_ids = train_df['SK_ID_CURR']
    test_ids = test_df['SK_ID_CURR']
    
    # Supprimer la cible et les identifiants des données d'entraînement et de test
    train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, inplace=True)
    test_df.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    print(f"Traitement des données avec {len(SELECTED_FEATURES)} caractéristiques sélectionnées...")
    
    # Vérifier quelles caractéristiques sont disponibles dans le jeu de données
    available_features = [f for f in SELECTED_FEATURES if f in train_df.columns]
    missing_features = [f for f in SELECTED_FEATURES if f not in train_df.columns]
    
    if missing_features:
        print(f"Attention: Les caractéristiques suivantes ne sont pas disponibles: {missing_features}")
    
    # Sélectionner uniquement les caractéristiques disponibles
    X_train = train_df[available_features].copy()
    X_test = test_df[available_features].copy()
    
    print(f"Prétraitement des données: gérer les valeurs manquantes et encoder les variables catégorielles...")
    
    # Prétraiter les deux ensembles de données
    X_train, X_test = preprocess_datasets(X_train, X_test)
    
    print(f"Forme finale des données d'entraînement: {X_train.shape}")
    print(f"Forme finale des données de test: {X_test.shape}")
    
    return X_train, X_test, y, train_ids, test_ids

def preprocess_datasets(train_df, test_df):
    """
    Prétraite les ensembles d'entraînement et de test séparément
    """
    # Fonction pour prétraiter un ensemble de données
    def preprocess_df(df):
        processed_df = df.copy()
        
        # Gérer les valeurs manquantes pour chaque colonne
        for col in processed_df.columns:
            # Si la colonne contient des valeurs manquantes
            if processed_df[col].isnull().sum() > 0:
                if processed_df[col].dtype == 'object':
                    # Pour les variables catégorielles, utiliser le mode
                    most_frequent = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "MISSING"
                    processed_df[col].fillna(most_frequent, inplace=True)
                else:
                    # Pour les variables numériques, utiliser la médiane
                    median_value = processed_df[col].median()
                    processed_df[col].fillna(median_value, inplace=True)
        
        # Encodage des variables catégorielles
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                le = LabelEncoder()
                processed_df[col] = processed_df[col].astype(str)  # Convertir en string pour éviter les problèmes
                le.fit(processed_df[col])
                processed_df[col] = le.transform(processed_df[col])
        
        # Vérification finale qu'il ne reste pas de NaN
        processed_df.fillna(0, inplace=True)  # Au cas où
        
        return processed_df
    
    # Prétraiter séparément les données d'entraînement et de test
    print("Prétraitement des données d'entraînement...")
    train_df_processed = preprocess_df(train_df)
    
    print("Prétraitement des données de test...")
    test_df_processed = preprocess_df(test_df)
    
    return train_df_processed, test_df_processed

def train_random_forest(X_train, y_train):
    """
    Entraîne un modèle Random Forest avec GridSearchCV pour trouver les meilleurs hyperparamètres
    """
    print("Entraînement du modèle Random Forest...")
    
    # Paramètres à tester
    param_grid = {
        'n_estimators': [100],  # Nombre d'arbres (réduit pour la rapidité)
        'max_depth': [10, 20],  # Profondeur maximale des arbres
        'min_samples_split': [5, 10],  # Nombre minimum d'échantillons pour diviser un nœud
        'min_samples_leaf': [2, 4]  # Nombre minimum d'échantillons dans une feuille
    }
    
    # Initialisation du modèle
    rf = RandomForestClassifier(random_state=42)
    
    # GridSearchCV pour trouver les meilleurs hyperparamètres
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # 3-fold CV
        scoring='roc_auc',  # Évaluation basée sur l'AUC
        n_jobs=-1,  # Utiliser tous les processeurs
        verbose=1
    )
    
    # Entraînement du modèle
    grid_search.fit(X_train, y_train)
    
    # Afficher les meilleurs hyperparamètres
    print("Meilleurs hyperparamètres: ", grid_search.best_params_)
    
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train):
    """
    Entraîne un modèle XGBoost avec GridSearchCV pour trouver les meilleurs hyperparamètres
    """
    print("Entraînement du modèle XGBoost...")
    
    # Paramètres à tester
    param_grid = {
        'n_estimators': [100],  # Nombre d'arbres (réduit pour la rapidité)
        'max_depth': [3, 5],  # Profondeur maximale des arbres
        'learning_rate': [0.1, 0.01],  # Taux d'apprentissage
        'subsample': [0.8, 1.0],  # Fraction d'échantillons à utiliser
        'colsample_bytree': [0.8, 1.0]  # Fraction de caractéristiques à utiliser
    }
    
    # Initialisation du modèle
    xgb_model = xgb.XGBClassifier(random_state=42)
    
    # GridSearchCV pour trouver les meilleurs hyperparamètres
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,  # 3-fold CV
        scoring='roc_auc',  # Évaluation basée sur l'AUC
        n_jobs=-1,  # Utiliser tous les processeurs
        verbose=1
    )
    
    # Entraînement du modèle
    grid_search.fit(X_train, y_train)
    
    # Afficher les meilleurs hyperparamètres
    print("Meilleurs hyperparamètres: ", grid_search.best_params_)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Évalue le modèle sur l'ensemble d'entraînement et de test
    """
    # Faire des prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilités pour AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculer les métriques
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Afficher les métriques
    print("\nMétriques d'évaluation:")
    print(f"Accuracy - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
    print(f"F1-Score - Train: {train_f1:.4f}, Test: {test_f1:.4f}")
    print(f"AUC - Train: {train_auc:.4f}, Test: {test_auc:.4f}")
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pas de défaut (0)', 'Défaut (1)'],
                yticklabels=['Pas de défaut (0)', 'Défaut (1)'])
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Calculer et afficher l'importance des caractéristiques
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances)
        plt.title('Importance des caractéristiques')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'feature_importances.png'))
        plt.close()
        
        print("\nImportance des caractéristiques:")
        print(feature_importances)
    
    return {
        'accuracy': test_accuracy,
        'f1': test_f1,
        'auc': test_auc
    }

def save_model(model, model_name):
    """
    Sauvegarde le modèle et ses métriques
    """
    # Créer le chemin du fichier
    model_file = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    
    # Sauvegarder le modèle
    joblib.dump(model, model_file)
    print(f"Modèle sauvegardé: {model_file}")
    
    # Ajouter un fichier texte avec la date de création et des informations sur le modèle
    with open(os.path.join(MODEL_DIR, f'{model_name}_info.txt'), 'w') as f:
        f.write(f"Modèle créé le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Type de modèle: {type(model).__name__}\n")
        if hasattr(model, 'get_params'):
            f.write(f"Paramètres du modèle:\n{model.get_params()}\n")

def main():
    """
    Fonction principale pour l'entraînement et l'évaluation des modèles
    """
    print("=== Début de l'entraînement du modèle ===")
    
    # Charger et prétraiter les données
    try:
        X_train_full, X_test_full, y_full, train_ids, test_ids = load_and_preprocess_data()
        
        # Diviser les données d'entraînement en ensembles d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )
        
        print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]} échantillons")
        print(f"Taille de l'ensemble de validation: {X_val.shape[0]} échantillons")
        
        # Entraîner les modèles
        rf_model = train_random_forest(X_train, y_train)
        xgb_model = train_xgboost(X_train, y_train)
        
        # Évaluer les modèles
        print("\nÉvaluation du modèle Random Forest:")
        rf_metrics = evaluate_model(rf_model, X_train, X_val, y_train, y_val)
        
        print("\nÉvaluation du modèle XGBoost:")
        xgb_metrics = evaluate_model(xgb_model, X_train, X_val, y_train, y_val)
        
        # Choisir le meilleur modèle en fonction de l'AUC
        if rf_metrics['auc'] > xgb_metrics['auc']:
            best_model = rf_model
            best_model_name = "model"
            print("\nLe modèle Random Forest est sélectionné comme le meilleur modèle.")
        else:
            best_model = xgb_model
            best_model_name = "model"
            print("\nLe modèle XGBoost est sélectionné comme le meilleur modèle.")
        
        # Ré-entraîner le meilleur modèle sur toutes les données d'entraînement
        print("\nRé-entraînement du meilleur modèle sur l'ensemble complet d'entraînement...")
        best_model.fit(X_train_full, y_full)
        
        # Sauvegarder le modèle
        save_model(best_model, best_model_name)
        
        # Sauvegarder également le modèle XGBoost pour compatibilité
        save_model(xgb_model, "xgboost_model")
        
        print("\n=== Fin de l'entraînement du modèle ===")
        
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 