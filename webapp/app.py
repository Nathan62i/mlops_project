from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
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

# Chemin du modèle MLflow
MODEL_PATH = "/model/sentiment-analyzer-model"

# Chargement du modèle MLflow
try:
    logger.info("Loading the MLflow model from {}", MODEL_PATH)
    model = mlflow.sklearn.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load the model: {}", e)
    raise RuntimeError("Model loading failed.") from e

@app.post("/predict", summary="Prédire les sentiments des critiques")
def predict(input: PredictInput):
    """
    Prend une liste de critiques (reviews) et renvoie leurs sentiments prédits.

    - **reviews** : Une liste de textes (critique) à analyser pour déterminer le sentiment.
    - Le modèle renvoie "positif" ou "negatif" pour chaque critique en fonction de son analyse.
    """
    logger.info("Received a prediction request with {} reviews.", len(input.reviews))
    
    try:
        # Création d'un DataFrame à partir des reviews d'entrée
        reviews = pd.DataFrame({'review': input.reviews})
        logger.debug("Input DataFrame: {}", reviews)

        # Prédictions avec le modèle
        predictions = model.predict(reviews["review"])  # Utilisez la méthode adaptée à votre modèle
        logger.debug("Raw predictions: {}", predictions)

        # Mappez les prédictions numériques en étiquettes lisibles (positif/negatif)
        sentiments = ["positif" if pred == 1 else "negatif" for pred in predictions]
        logger.debug("Mapped sentiments: {}", sentiments)

        logger.info("Prediction request processed successfully.")
        return {"sentiments": sentiments}
    except Exception as e:
        logger.error("An error occurred during prediction: {}", e)
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.") from e

