import argparse
import os
import random

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np

from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_NAME = "Iris_LogisticRegression_Experiments"
TRACKING_URI = "http://127.0.0.1:5000"


def parse_arguments():
    """
    Define los argumentos que pueden modificarse
    desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento parametrizado de Logistic Regression sobre Iris."
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización. Default: 1.0"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Número máximo de iteraciones. Default: 200"
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerancia para el criterio de convergencia. Default: 1e-4"
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        help="Solver utilizado por Logistic Regression. Default: lbfgs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad. Default: 42"
    )

    return parser.parse_args()


def set_seed(seed):
    """
    Fija las semillas utilizadas durante la ejecución.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    """
    Carga el dataset Iris y separa variables predictoras
    y variable objetivo.
    """
    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    return X, y, iris.target_names


def split_data(X, y, seed):
    """
    Divide los datos en entrenamiento y prueba,
    manteniendo el balance entre clases.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_model(C, max_iter, tol, solver, seed):
    """
    Construye el pipeline de preprocesamiento
    y clasificación.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                tol=tol,
                solver=solver,
                random_state=seed
            )
        )
    ])

    return model


def evaluate_model(model, X_test, y_test):
    """
    Genera predicciones y calcula las métricas principales.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    return y_pred, accuracy, f1_macro


def save_confusion_matrix(y_test, y_pred, target_names):
    """
    Genera la matriz de confusión y la guarda
    como artefacto local.
    """
    os.makedirs("artifacts", exist_ok=True)

    artifact_path = "artifacts/confusion_matrix.png"

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )

    disp.plot()

    plt.title("Matriz de confusión - Iris")
    plt.tight_layout()
    plt.savefig(artifact_path)
    plt.close()

    return artifact_path


def run_experiment(args):
    """
    Ejecuta el flujo completo de entrenamiento,
    evaluación y registro en MLflow.
    """

    # Reproducibilidad
    set_seed(args.seed)

    # Datos
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        args.seed
    )

    # Modelo
    model = build_model(
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
        solver=args.solver,
        seed=args.seed
    )

    model.fit(X_train, y_train)

    # Evaluación
    y_pred, accuracy, f1_macro = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Artefacto
    confusion_matrix_path = save_confusion_matrix(
        y_test,
        y_pred,
        target_names
    )

    # Conexión con MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = (
        f"C_{args.C}"
        f"_iter_{args.max_iter}"
        f"_tol_{args.tol}"
        f"_{args.solver}"
    )

    with mlflow.start_run(run_name=run_name) as run:

        # Parámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("tol", args.tol)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", 0.20)

        # Métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        # Artefacto
        mlflow.log_artifact(confusion_matrix_path)

        # Firma del modelo
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # Modelo entrenado
        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_logistic_regression",
            signature=signature,
            input_example=X_train.head()
        )

        print("\nCorrida registrada correctamente")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"C: {args.C}")
        print(f"max_iter: {args.max_iter}")
        print(f"tol: {args.tol}")
        print(f"solver: {args.solver}")
        print(f"seed: {args.seed}")


def main():
    args = parse_arguments()
    run_experiment(args)


if __name__ == "__main__":
    main()
