import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

print("🔍 Chargement des données...")
# Créer dossier modèle s'il n'existe pas
os.makedirs('./model', exist_ok=True)

# Chargement des données
train_data = pd.read_csv('./data/application_train.csv', encoding='utf-8')

print("🔍 Imputation des valeurs manquantes...")
# Imputation des valeurs manquantes
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    train_data[col].fillna(train_data[col].mean(), inplace=True)

categorical_columns = train_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

print("🔍 Encodage des variables catégorielles...")
# Encodage des variables catégorielles
label_encoder = LabelEncoder()
for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col].astype(str))

print("🔍 Sélection des features...")
# Sélection des features
features = [
    'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH',
    'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'
]
X = train_data[features]
y = train_data['TARGET']

print("🔍 Standardisation des features...")
# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔍 Rééquilibrage des classes...")
# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("🔍 Split des données...")
# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("🔍 Entrainement du modèle Random Forest...")
# ======================
# Random Forest
# ======================
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf)
print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("🔍 Entrainement du modèle XGBoost...")
# ======================
# XGBoost
# ======================
xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
f1_xgb = f1_score(y_test, y_pred_xgb)
print("=== XGBoost ===")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# ======================
# Neural Network
# ======================
# nn_model = Sequential([
#     Dense(128, activation='relu', input_dim=X_train.shape[1]),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
# ])
# nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
# f1_nn = f1_score(y_test, y_pred_nn)
# print("=== Neural Network ===")
# print(classification_report(y_test, y_pred_nn))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn))

# ======================
# Sélection et sauvegarde du meilleur modèle
# ======================
print("🔍 Sélection et sauvegarde du meilleur modèle...")
models = {'random_forest': (rf_model, f1_rf), 'xgboost': (xgb_model, f1_xgb)} # , 'neural_network': (nn_model, f1_nn)
best_model_name, (best_model, best_f1) = max(models.items(), key=lambda x: x[1][1])
print(f"\n✅ Meilleur modèle : {best_model_name} avec F1-score = {best_f1:.4f}")

if best_model_name in ['random_forest', 'xgboost']:
    joblib.dump(best_model, './model/best_model.pkl')
# else:
#     nn_model.save('./model/best_model_nn.h5')

# ======================
# Matrices de confusion en image
# ======================
print("🔍 Matrices de confusion en image...")
def plot_conf_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.savefig(f'./model/confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()

print("🔍 Sauvegarde des matrices de confusion en image...")
plot_conf_matrix(confusion_matrix(y_test, y_pred_rf), "Random Forest")
plot_conf_matrix(confusion_matrix(y_test, y_pred_xgb), "XGBoost")
# plot_conf_matrix(confusion_matrix(y_test, y_pred_nn), "Neural Network")
