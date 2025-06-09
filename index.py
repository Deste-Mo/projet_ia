import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import os

# Charger les données
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    data = pd.read_csv(file_path)
    X = data.drop('achete', axis=1)  # Supprime la colonne 'achete'
    y = data['achete']  # Prend la colonne 'achete' comme cible
    return X, y

# Diviser les données
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Préparer les données
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Créer le modèle
def build_model(input_shape):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),  # Couche d'entrée
        Dense(8, activation='relu'),  # Couche cachée
        Dense(1, activation='sigmoid')  # Couche de sortie pour classification binaire
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entraîner le modèle
def train_model(model, X_train, y_train, epochs=10, batch_size=2, validation_split=0.2):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, verbose=1)
    return history

# Évaluer le modèle
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Perte sur le test : {loss:.4f}")
    print(f"Précision sur le test : {accuracy:.4f}")
    
    # Prédictions et rapport
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

# Sauvegarder le modèle
def save_model(model, file_path):
    model.save(file_path)
    print(f"Modèle sauvegardé à : {file_path}")

def main():
    # Chemin du fichier CSV
    file_path = r'C:\Users\Deste MO\Documents\ia\projet\data.csv'
    
    try:
        # Charger les données
        X, y = load_data(file_path)
        
        # Diviser les données
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Préparer les données
        X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
        
        # Créer le modèle
        model = build_model(X_train.shape[1])
        
        # Entraîner le modèle
        train_model(model, X_train_scaled, y_train)
        
        # Évaluer le modèle
        evaluate_model(model, X_test_scaled, y_test)
        
        # Sauvegarder le modèle
        save_model(model, r'C:\Users\Deste MO\Documents\ia\modele_achat.h5')
        
    except Exception as e:
        print(f"Erreur : {str(e)}")

if __name__ == "__main__":
    main()