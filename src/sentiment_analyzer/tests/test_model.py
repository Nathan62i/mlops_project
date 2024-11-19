import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, recall_score
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import unittest

mlflow.set_tracking_uri("http://localhost:5000")

TEST_MODEL_NAME = "logisticRegression"
TEST_MODEL_VERISON = 12
TEST_TEST_SET = 0.60

model = mlflow.sklearn.load_model(model_uri = f"models:/{TEST_MODEL_NAME}/{TEST_MODEL_VERISON}")

# Fonction de test
def test_entree_simple():
    X = pd.DataFrame({
    'review': [
        "bonjour je suis Amadou",
        "le film était nul en plus j'ai des bonbons",
        "HAHAHAHAHAHAHAHHA" ]})
    
    # Faire des prédictions
    y = model.predict(X["review"])
    
    # Vérifier que toutes les valeurs prédites sont dans {0, 1}
    unique_values = np.unique(y)
    if not set(unique_values).issubset({0, 1}):
        raise AssertionError(f"Les prédictions contiennent des valeurs inattendues : {unique_values}")
    print("Test réussi : test_entree_simple.")

def test_entree_unusuelle():
    X2 = pd.DataFrame({
    'review': [
        "",
        "..@@@"]})
    
    # Faire des prédictions
    y = model.predict(X2["review"])
    
    # Vérifier que toutes les valeurs prédites sont dans {0, 1}
    unique_values = np.unique(y)
    if not set(unique_values).issubset({0, 1}):
        raise AssertionError(f"Les prédictions contiennent des valeurs inattendues : {unique_values}")
    print("Test réussi : test_entree_unusuelle.")
    
def test_resultat_entendu():
    X3 = pd.DataFrame({
    'review': [
        "Beur sur la ville réunit à lui même toutes les lacunes d'un film français médiocre. Les acteurs sont amateurs, nuls, moches, beurs et surement handicapés mentalement (voir plus pour certains). Le scénario est... où??? J'ai vu ce film cela fait trois semaines, la seule chose dont je me rappelle est de lui mettre 0/20. Une honte à nôtre pays! ----Novembre 2013----",
        "Premier film de la saga Kozure Okami, \"Le Sabre de la vengeance\"est un très bon film qui mêle drame et action, et qui, en 40 ans, n'a pas pris une ride."]})
    
    # Faire des prédictions
    y = model.predict(X3["review"])
    
    # Vérifier que toutes les valeurs prédites sont dans {0, 1}
    unique_values = np.unique(y)
    if not set(unique_values).issubset({0, 1}):
        raise AssertionError(f"Les prédictions contiennent des valeurs inattendues : {unique_values}")
    
    if len(y) >= 2 and y[0] == 0 and y[1] == 1:
        print("Test réussi : test_resultat_entendu")
    else:
        raise AssertionError(f"Les prédictions ne suivent pas l'ordre attendu : {y}")

def test_accuracy():
    X4 = pd.DataFrame({
    'review': [
        "Beur sur la ville réunit à lui même toutes les lacunes d'un film français médiocre. Les acteurs sont amateurs, nuls, moches, beurs et surement handicapés mentalement (voir plus pour certains). Le scénario est... où??? J'ai vu ce film cela fait trois semaines, la seule chose dont je me rappelle est de lui mettre 0/20. Une honte à nôtre pays! ----Novembre 2013----",
        "Premier film de la saga Kozure Okami, \"Le Sabre de la vengeance\"est un très bon film qui mêle drame et action, et qui, en 40 ans, n'a pas pris une ride.",
        "Un rythme bien trop lent et un Ashton Kutcher bien trop nul!"]})
    
    y4 = np.array([0, 1, 0])
    
    y_pred = model.predict(X4["review"])  # Prédictions du modèle
    
    # Calculer l'accuracy
    accuracy = accuracy_score(y4, y_pred)
    
    if accuracy > TEST_TEST_SET:
        print("Test réussi : test_accuracy")
    else:
        raise AssertionError(f"L'accuracy n'est pas celle attendu")
                        