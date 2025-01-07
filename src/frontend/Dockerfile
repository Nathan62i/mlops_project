# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY app.py ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de Streamlit
EXPOSE 8501

# Commande pour exécuter Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

