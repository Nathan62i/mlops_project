import click
import pandas as pd
from .tests import test_model

@click.command()
@click.option('--input_file', type=str, default=None, help='Fichier CSV contenant les messages à traiter.')
@click.option('--output_file', type=str, default=None, help='Fichier CSV de sortie pour les prédictions.')
@click.option('--text', type=str, default=None, help='Texte unique à analyser.')
@click.option('--model_name', type=str, required=True, help='Nom du modèle à utiliser.')
@click.option('--model_version', type=int, required=True, help='Version du modèle à utiliser.')
@click.option('--mlflow_url', type=str, default="http://localhost:5000", help='URL du serveur MLFlow.')
def main(input_file, output_file, text, model_name, model_version, mlflow_url):
    # Vérification des arguments
    if not (input_file or text):
        raise ValueError("Vous devez fournir soit --input_file, soit --text.")
    
    # Création de l'instance de ModelManager
    model_manager = ModelManager(model_name, model_version, mlflow_url)
    
    # Chargement du modèle
    model_manager.load_model()

    # Cas 1 : Traitement d'un fichier CSV
    if input_file:
        df = pd.read_csv(input_file)
        df = model_manager.predict(df)
        if output_file:
            model_manager.save_predictions(df, output_file)
        else:
            print(df)

    # Cas 2 : Traitement d'un texte unique
    elif text:
        prediction = model_manager.predict_text(text)
        print(f"La polarité du texte est : {prediction}")

if __name__ == '__main__':
    main()


test_result = subprocess.run(
    ["pytest",pkg_resources.resource_filename('src', "./tests/test_model.py")], capture_output=False)