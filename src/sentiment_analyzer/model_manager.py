import mlflow
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class ModelManager:
    def __init__(self, model_name, model_version, mlflow_url="http://localhost:5000"):
        self.model_name = model_name
        self.model_version = model_version
        self.mlflow_url = mlflow_url
        self.model = None
        mlflow.set_tracking_uri(self.mlflow_url)

    def load_model(self):
        """Charge le modèle depuis MLFlow registry."""
        print(f"Loading model: {self.model_name}, version: {self.model_version}")
        self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{self.model_version}")
    
    def predict(self, text_or_file):
        """Prédire la polarité du texte ou des messages dans un fichier."""
        if isinstance(text_or_file, str):
            return self.predict_text(text_or_file)
        elif isinstance(text_or_file, pd.DataFrame):
            return self.predict_file(text_or_file)
    
    def predict_text(self, text):
        """Prédire la polarité d'un texte unique."""
        reviews = pd.DataFrame({'review': [text]})
        return self.model.predict(reviews["review"])

    def predict_file(self, df):
        """Prédire la polarité de tous les textes dans un fichier."""
        reviews = df['review']
        predictions = self.model.predict(reviews)
        return predictions

