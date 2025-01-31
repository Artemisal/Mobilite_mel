import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import mlflow
import mlflow.sklearn
import numpy as np

# Charger les données
file_path = 'data/finalData.csv'  # Change the path if necessary
data = pd.read_csv(file_path)

# Définir les features et la variable cible
X = data[['P7', 'P10','PNPC','bike_freq', 'car_freq', 'public_freq', 'AgeGroup','D5AA', 'D2AA', 'speed', 'D8C', 'DIST_km','D4A']]
X = sm.add_constant(X)
y = data['ECO']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Début de l'expérience MLflow
mlflow.set_experiment("ECO Logistic Regression")

with mlflow.start_run(run_name="Logistic Regression Model"):
    # Initialiser et entraîner le modèle
    logit_model = sm.Logit(y_train, X_train)  # Logistic regression
    result = logit_model.fit()  # Fitting the model

    # Prédictions
    y_pred_proba = result.predict(X_test)  # Predicted probabilities
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Enregistrer les paramètres
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Enregistrer les métriques
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R-squared", r2)

    # Enregistrer le modèle
    model_path = "logistic_regression_model"
    mlflow.sklearn.log_model(result, model_path)  # Save the model
    print(f"Modèle enregistré dans MLflow : {model_path}")

    # Enregistrer le résumé du modèle comme artefact
    with open("model_summary.txt", "w") as f:
        f.write(result.summary().as_text())
    mlflow.log_artifact("model_summary.txt")

    # Afficher les métriques
    print(f"Accuracy: {accuracy}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")

    print("Résumé du modèle enregistré dans MLflow.")