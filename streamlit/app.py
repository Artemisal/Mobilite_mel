import streamlit as st
import mlflow.pyfunc
import pandas as pd

# Load models using MLflow
LOGISTIC_REG_URI = 'runs:/bf763598d5134164b6035946c899e983/logistic_regression_model'
RF_REG_URI = 'runs:/2801155d596c478db2be457366b859d3/random_forest_model'  # Replace with your actual Random Forest model URI

logistic_reg_model = mlflow.pyfunc.load_model(LOGISTIC_REG_URI)
random_forest_model = mlflow.pyfunc.load_model(RF_REG_URI)

#CSS
st.markdown(
    """
    <style>
    body {
         background-image: url("https://images.unsplash.com/photo-1626314904811-46de72506c22?q=80&w=1780&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;  /* Ensures the image covers the entire background */
        background-position: center center;  /* Centers the image both vertically and horizontally */
        background-repeat: no-repeat;  /* Prevents the image from repeating */
        background-attachment: fixed;  /* Keeps the background fixed during scroll */
        height: 100vh;  /* Makes the background take up the full viewport height */
        overflow: auto;
    }

    .stApp {
        background: rgba(255, 255, 255, 0.9);  /* Transparence pour améliorer la lisibilité */
        padding: 30px;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
        font-family: 'Helvetica', sans-serif;
    }

    h1 {
        color: #4CAF50;
        text-align: center;
    }

    .stButton>button {
        background-color: #4CAF50; /* Vert par défaut */
        color: white;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
        transition: background-color 0.3s; 
    }

    .stButton>button:hover {
        background-color: #45a049; 
    }

    .stTextInput, .stNumberInput {
        width: 100%;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for model selection
st.sidebar.header('Select Model')
model_choice = st.sidebar.selectbox('Choose Model', ['Logistic Regression', 'Random Forest'])

# Mapping D5AA and D2AA (Motif à la destination and Motif à l'origine) to numeric values
motif_map = {
    'home': 1,
    'work': 2,
    'education': 3,
    'shop': 4,
    'leisure': 5
}
p7_map = {
    'Travail à plein temps': 1,
    'Travail à temps partiel': 2,
    'Apprentissage, formation, stage': 3,
    'Étudiant': 4,
    'Scolaire jusqu\'au bac': 5,
    'Chômeur, recherche un emploi': 6,
    'Retraité': 7,
    'Reste au foyer': 8,
    'Autre': 9,
    'Non réponse': 0
}

pnpc_map = {
    '1': 1,  # 1 PERSONNE
    '2': 2,  # 2 PERSONNES
    '3': 3,  # 3 PERSONNES
    '4': 4,  # 4 PERSONNES
    '5': 5,  # 5 PERSONNES
    '6': 6   # 6 PERSONNES OU PLUS
}

# P10 mapping
p10_map = {
    'oui': 1,  # OUI
    'non': 0   # NON
}
st.title("Prediction transport type 🚇")
st.write(
        """
        Veuillez entrer les informations nécessaires pour prédire votre utilisation des transports.
        """
)

with st.form(key="prediction_form"):
    # Collecting input data
    P7 = st.selectbox('Occupation principale', list(p7_map.keys()))
    P7 = p7_map[P7]

    P10 = st.selectbox('Abonnement Transport en commun:', list(p10_map.keys()))
    P10 = p10_map[P10]

    PNPC = st.selectbox('Taille de ménage:', list(pnpc_map.keys()))
    PNPC = pnpc_map[PNPC]

    # Utilisation du vélo
    bike_freq = st.selectbox('Utilisation du vélo', ['Non spécifié', 'Fréquemment', 'Pas fréquemment'])
    bike_freq = None if bike_freq == 'Non spécifié' else (1 if bike_freq == 'Fréquemment' else 0)

    # Utilisation de la voiture
    car_freq = st.selectbox('Utilisation de la voiture', ['Non spécifié', 'Fréquemment', 'Pas fréquemment'])
    car_freq = None if car_freq == 'Non spécifié' else (1 if car_freq == 'Fréquemment' else 0)

    # Utilisation des transports en commun
    public_freq = st.selectbox('Utilisation des transports en commun',
                               ['Non spécifié', 'Fréquemment', 'Pas fréquemment'])
    public_freq = None if public_freq == 'Non spécifié' else (1 if public_freq == 'Fréquemment' else 0)

    D5AA = st.selectbox('Motif à la destination (D5AA)', ['home', 'work', 'education', 'shop', 'leisure'])
    D2AA = st.selectbox('Motif à l\'origine (D2AA)', ['home', 'work', 'education', 'shop', 'leisure'])

    # Age group mapping
    AgeGroup = st.selectbox(
        'Age Group',
        ['Inactifs', 'Retraités', 'Actifs'],
        index=0
    )
    age_group_mapping = {'Inactifs': 0, 'Retraités': 1, 'Actifs': 2}
    AgeGroup = age_group_mapping[AgeGroup]

    speed = st.number_input('Vitesse moyenne', min_value=0.0)
    D8C = st.number_input('Durée du déplacement (minutes)', min_value=0.0)
    DIST_km = st.number_input('Distance (km)', min_value=0.0, max_value=1000.0, value=100.0)
    D4A = st.number_input('Heure de départ', min_value=0)

    # Submit button
    submit_button = st.form_submit_button(label='Get Prediction')

if submit_button:
    # Prepare input data as a DataFrame
    if bike_freq is None:
        bike_freq = 0
    if car_freq is None:
        car_freq = 0  # Idem
    if public_freq is None:
        public_freq = 0

    input_data = {
        "P7": P7,
        "P10": P10,
        "PNPC": PNPC,
        "bike_freq": bike_freq,
        "car_freq": car_freq,
        "public_freq": public_freq,
        "D5AA": motif_map[D5AA],
        "D2AA": motif_map[D2AA],
        "AgeGroup": AgeGroup,
        "speed": speed,
        "D8C": D8C,
        "DIST_km": DIST_km,
        "D4A": D4A,
    }



    input_df = pd.DataFrame([input_data])

    # Ensure all columns are numeric (for Logistic Regression and Random Forest models)
    input_df = input_df.apply(pd.to_numeric,
                              errors='coerce')  # Convert all columns to numeric, non-numeric will be set to NaN

    # Add a constant column for logistic regression intercept if selected
    if model_choice == "Logistic Regression":
        input_df.insert(0, "const", 1)  # Add constant column for logistic regression
        # Check if there are any NaN values after conversion
        if input_df.isnull().values.any():
            st.error("There are invalid input values (NaN detected). Please correct your inputs.")
        else:
            # Predict using Logistic Regression
            prediction = logistic_reg_model.predict(input_df)
            # Convert probability to binary class (0 or 1)
            prediction = 1 if prediction[0] >= 0.5 else 0  # Threshold at 0.5 for binary classification
    elif model_choice == "Random Forest":
        # Predict using Random Forest
        prediction = random_forest_model.predict(input_df)

        # Change the background image based on the predicted class
        if prediction[0] == 'car':  # If the model predicts 'car'
            st.markdown(
                """
                <style>
                body {
                    background-image: url('https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        elif prediction[0] == 'pt':  # If the model predicts 'car'
            st.markdown(
                """
                <style>
                body {
                    background-image: url('https://images.unsplash.com/photo-1603939540289-922b5886c597?q=80&w=2050&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    # Display prediction result
    st.write(f'Prediction: {prediction}')  # Display 0 or 1 for logistic regression