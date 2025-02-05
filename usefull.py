import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from eli5.sklearn import PermutationImportance
from sklearn.inspection import PartialDependenceDisplay

# La plupart des fonctions suivantes sont des fonctions que j'ai du coder pour un devoir d'un autre cours

def split_data(
    data: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """
    Divise les données en ensembles d'entraînement et de test.

    Paramètres:
    - data (pd.DataFrame): Ensemble de données d'entrée.
    - test_size (float): Proportion de l'ensemble de données à inclure dans la division de test.

    Renvoie:
    - tuple: Un tuple contenant les DataFrames X_train, X_test, y_train et y_test.
    """
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, max_depth: int = None, random_state: int = 42,
) -> RandomForestClassifier:
    """
    Entraîne un classificateur Random Forest.

    Paramètres:
    - X_train (pd.DataFrame): Caractéristiques de l'ensemble d'entraînement.
    - y_train (pd.Series): Étiquettes cibles de l'ensemble d'entraînement.
    - n_estimators (int): Nombre d'arbres dans la forêt.
    - max_depth (int): Profondeur maximale des arbres de la forêt (par défaut=None).

    Renvoie:
    - RandomForestClassifier: Modèle Random Forest entraîné.
    """
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    random_forest.fit(X_train, y_train)

    return random_forest

def predict_random_forest(model, X_features):
    """
    Effectue des prédictions avec un modèle Random Forest sur un ensemble de caractéristiques.

    Paramètres:
    - model (RandomForestClassifier): Le modèle Random Forest entraîné.
    - X_features (pd.DataFrame): Les caractéristiques pour lesquelles faire des prédictions.

    Renvoie:
    - pd.DataFrame: Un DataFrame contenant les prédictions.
    """
    predictions = model.predict(X_features)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Class'])
    
    return predictions_df

def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple:
    """
    Évalue le modèle Random Forest.

    Paramètres:
    - model (RandomForestClassifier): Modèle Random Forest entraîné.
    - X_test (pd.DataFrame): Caractéristiques de l'ensemble de test.
    - y_test (pd.Series): Étiquettes cibles de l'ensemble de test.

    Renvoie:
    - tuple: Un tuple contenant la précision (float) et le rapport de classification (str).
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    classif_report = classification_report(y_test, y_pred)

    return (accuracy, classif_report)

def calculate_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_sate: int = 1,
):
    """
    Calcule les importances par permutation pour un modèle d'apprentissage automatique.

    Paramètres:
    - model: Modèle d'apprentissage automatique entraîné.
    - X_val: Caractéristiques de l'ensemble de validation.
    - y_val: Étiquettes cibles de l'ensemble de validation.

    Renvoie:
    - eli5.PermutationImportance: Objet PermutationImportance avec les importances calculées. Nous n'utiliserons que le modèle et la valeur prédéfinie pour random_state.
    """

    perm = PermutationImportance(model, random_state = random_sate).fit(X_val, y_val)

    return perm

def plot_partial_dependence(model, X_val: pd.DataFrame, feature_name: str):
    """
    Affiche les tracés de dépendance partielle pour une caractéristique spécifiée.

    Paramètres:
    - model: Modèle d'apprentissage automatique entraîné.
    - X_val: Caractéristiques de l'ensemble de validation.
    - feature_name: Nom de la caractéristique pour laquelle créer les tracés de dépendance partielle.
    """
    pdp_display = PartialDependenceDisplay.from_estimator(model, X_val, features=[feature_name])
    
    pdp_display.figure_.suptitle(f"Tracé de Dépendance Partielle pour {feature_name}")

    plt.grid(True)