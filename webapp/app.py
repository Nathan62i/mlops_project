from fastapi import FastAPI
from pydantic import BaseModel
import os
import mlflow.sklearn
import pandas as pd

# Initialisation de l'application FastAPI avec une description et version
app = FastAPI(
    title="Sentiment Analysis API",
    description="API pour analyser le sentiment de critiques (reviews) en utilisant un modèle de sentiment.",
    version="1.0.0"
)

# Définition du schéma d'entrée pour la prédiction
class PredictInput(BaseModel):
    reviews: list[str]  # Liste de textes à analyser

# Chargement du modèle MLflow
MODEL_PATH = "/model"  # Assurez-vous que le modèle est stocké dans ce répertoire

# Charger le modèle MLflow à partir du chemin spécifié
model = mlflow.sklearn.load_model(MODEL_PATH)

@app.post("/predict", summary="Prédire les sentiments des critiques")
def predict(input: PredictInput):
    """
    Prend une liste de critiques (reviews) et renvoie leurs sentiments prédits.

    - **reviews** : Une liste de textes (critique) à analyser pour déterminer le sentiment.
    - Le modèle renvoie "positif" ou "negatif" pour chaque critique en fonction de son analyse.
    """

    # Création d'un DataFrame à partir des reviews d'entrée
    reviews = pd.DataFrame({'review': input.reviews})

    # Prédictions avec le modèle
    predictions = model.predict(reviews["review"])  # Utilisez la méthode adaptée à votre modèle

    # Mappez les prédictions numériques en étiquettes lisibles (positif/negatif)
    sentiments = ["positif" if pred == 1 else "negatif" for pred in predictions]

    return {"sentiments": sentiments}

