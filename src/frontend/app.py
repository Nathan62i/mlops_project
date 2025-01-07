import streamlit as st
import requests

# URL de votre API backend (à ajuster si nécessaire)
API_URL = "http://webapp:8000/predict"

# Titre de l'application
st.header("Sentiment Analysis Frontend")

# Zone de texte pour l'utilisateur
user_input = st.text_area("Entrez vos critiques (une par ligne) :", height=200)

# Bouton pour lancer la prédiction
if st.button("Analyser les sentiments"):
    if user_input.strip():  # Vérifiez que l'entrée n'est pas vide
        # Préparez les données à envoyer
        reviews = user_input.strip().split("\n")
        payload = {"reviews": reviews}

        try:
            # Appel de l'API backend
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                # Affichez les résultats si tout s'est bien passé
                sentiments = response.json().get("sentiments", [])
                st.success("Résultats des prédictions :")
                for review, sentiment in zip(reviews, sentiments):
                    st.write(f"- **Critique** : {review}")
                    st.write(f"  **Sentiment** : {sentiment}")
            else:
                st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
    else:
        st.warning("Veuillez entrer au moins une critique avant de lancer la prédiction.")
