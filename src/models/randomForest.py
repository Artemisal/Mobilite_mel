import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn

# Charger les données
file_path = 'data/finalData.csv'
deplacement = pd.read_csv(file_path)

# Définir les variables indépendantes (X) et la variable dépendante (y)
X = deplacement[['P7', 'P10','PNPC','bike_freq', 'car_freq', 'public_freq','D5AA', 'D2AA', 'AgeGroup', 'speed', 'D8C', 'DIST_km', 'D4A']]
y = deplacement['MODP']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Début de l'expérience MLflow
mlflow.set_experiment("Random Forest for MODP")

with mlflow.start_run(run_name="Random Forest Model"):
    # Initialiser le modèle RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Entraîner le modèle avec les données d'entraînement
    rf_model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = rf_model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Afficher les résultats d'évaluation
    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # Enregistrer les paramètres
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("class_weight", "balanced")

    # Enregistrer les métriques
    mlflow.log_metric("Accuracy", accuracy)

    # Enregistrer la matrice de confusion comme artefact
    conf_matrix_df = pd.DataFrame(conf_matrix, index=rf_model.classes_, columns=rf_model.classes_)
    conf_matrix_path = "confusion_matrix.csv"
    conf_matrix_df.to_csv(conf_matrix_path)
    mlflow.log_artifact(conf_matrix_path)

    # Enregistrer le rapport de classification comme artefact
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_path = "classification_report.csv"
    class_report_df.to_csv(class_report_path)
    mlflow.log_artifact(class_report_path)

    # Enregistrer le modèle
    model_path = "random_forest_model"
    mlflow.sklearn.log_model(rf_model, model_path)
    print(f"Modèle enregistré dans MLflow : {model_path}")