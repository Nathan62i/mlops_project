# Étape 1 : Partir d'une image légère
FROM python:3.10-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /webapp

# Étape 3 : Installer les dépendances génériques
RUN pip install --no-cache-dir mlflow fastapi uvicorn

# Étape 4 : Copier les fichiers source dans l'image
COPY get_mlflow_model.py app.py /webapp/

# Étape 5 : Récupérer le modèle depuis le serveur MLFlow
# Utilisation des arguments passés lors de la construction
ARG MODEL_PATH
COPY $MODEL_PATH /model

# Étape 6 : Installer les dépendances spécifiques au modèle
RUN pip install --no-cache-dir -r /model/requirements.txt

# Étape 7 : Exposer le port de l'application
EXPOSE 8000

# Étape 8 : Commande par défaut pour lancer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
