# Projet de Classification Binaire avec TensorFlow

Ce projet implémente un modèle de réseau de neurones simple pour prédire si un client va **acheter** (`achete` = 1) ou non (`achete` = 0), à partir d’un jeu de données au format CSV.

---

## Description

Le modèle est construit avec TensorFlow et Keras. Il s'agit d'un réseau de neurones avec deux couches cachées, utilisé pour une classification binaire. Le pipeline comprend le chargement des données, la normalisation, la division en ensembles d'entraînement et de test, l'entraînement du modèle, l'évaluation et la sauvegarde du modèle entraîné.

---

## Prérequis

- Python 3.7+
- TensorFlow
- Pandas
- Scikit-learn

Tu peux installer les dépendances avec :

```bash
pip install tensorflow pandas scikit-learn
```

---

## Structure du projet

- `main.py` : Script principal contenant tout le code.
- `data.csv` : Fichier CSV contenant les données d'entrée. Ce fichier doit contenir une colonne cible nommée `achete` et les autres colonnes correspondant aux features.

---

## Utilisation

1. Placer le fichier `data.csv` au chemin défini dans le script (ou modifier le chemin dans le code).
2. Exécuter le script :

```bash
python main.py
```

3. Le modèle sera entraîné et évalué. Les métriques de performance seront affichées dans la console.
4. Le modèle entraîné sera sauvegardé au chemin `modele_achat.h5`.

---

## Fonctionnalités principales

- Chargement et préparation des données
- Standardisation des features
- Construction d’un modèle Keras séquentiel
- Entraînement avec validation croisée
- Évaluation avec rapport de classification (précision, rappel, F1-score)
- Sauvegarde du modèle pour une utilisation ultérieure

---

## Personnalisation

- Modifier le chemin des fichiers dans `main.py` pour adapter à votre environnement.
- Modifier les hyperparamètres (nombre d'époques, taille du batch, architecture du réseau) dans les fonctions correspondantes.

---

## Exemple de données

Le fichier CSV doit contenir au minimum une colonne cible nommée `achete` (valeurs 0 ou 1), ainsi que des colonnes numériques ou catégoriques (préalablement transformées) pour les features.

---

## Contact

Pour toute question ou suggestion, contactez-moi.

---

**Auteur :** Modeste  
**Date :** 2025
