# Experimentación con MLflow - Iris

Actividad individual de la materia **Operaciones de Aprendizaje Automático (MLOps)**.
Alumno: Diego Elian Rodriguez Cantu A00829925

El proyecto convierte el modelo baseline desarrollado previamente en un notebook a un script de entrenamiento parametrizado y reproducible utilizando MLflow Tracking.

## Dataset

Se utiliza el dataset Iris incluido en Scikit-learn.

El problema consiste en clasificar flores en tres especies:

- Setosa
- Versicolor
- Virginica

a partir de cuatro características:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Modelo

El modelo utiliza un pipeline compuesto por:

1. StandardScaler
2. LogisticRegression

## Parámetros disponibles

El script permite modificar desde línea de comandos:

- `--C`: inverso de la fuerza de regularización
- `--max-iter`: número máximo de iteraciones
- `--tol`: tolerancia de convergencia
- `--solver`: algoritmo de optimización
- `--seed`: semilla aleatoria

Los argumentos cuentan con valores por defecto.

## Instalación

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
