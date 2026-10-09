# Experimentación con MLflow - Iris

Actividad individual de la materia **Operaciones de Aprendizaje Automático (MLOps)**.  
**Alumno:** Diego Elian Rodriguez Cantu - A00829925

Este proyecto convierte el modelo baseline desarrollado previamente en un notebook a un script de entrenamiento parametrizado y reproducible utilizando MLflow Tracking.

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

1. `StandardScaler`
2. `LogisticRegression`

El código de entrenamiento se encuentra en `train.py` y está organizado en funciones para la carga de datos, división de entrenamiento y prueba, construcción del modelo, evaluación y registro del experimento en MLflow.

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/DiegoElian02/A00829925-mlops-iris-experimentation.git
cd A00829925-mlops-iris-experimentation
```

### 2. Crear un entorno virtual

Se recomienda utilizar Python 3.10 o superior.

En Linux, WSL o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las versiones utilizadas durante el desarrollo para facilitar la reproducibilidad del experimento.

### 4. Iniciar MLflow Tracking

Antes de ejecutar el script debe iniciarse el servidor local de MLflow.

En una terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Esta terminal debe mantenerse abierta mientras se ejecutan los experimentos.

La interfaz gráfica de MLflow estará disponible en:

```text
http://localhost:5000
```

### 5. Ejecutar el script

En una segunda terminal, con el entorno virtual activado, se puede ejecutar el modelo utilizando los parámetros por defecto:

```bash
python train.py
```

También es posible modificar los hiperparámetros directamente desde la línea de comandos sin editar el código.

Ejemplo:

```bash
python train.py --C 10 --max-iter 200 --tol 0.0001 --solver lbfgs --seed 42
```

Este fue uno de los comandos utilizados durante la experimentación.

Para consultar todos los argumentos disponibles:

```bash
python train.py --help
```

## Parámetros disponibles

| Argumento | Descripción | Valor por defecto |
|---|---|---|
| `--C` | Inverso de la fuerza de regularización | `1.0` |
| `--max-iter` | Número máximo de iteraciones | `200` |
| `--tol` | Tolerancia utilizada para la convergencia | `0.0001` |
| `--solver` | Algoritmo de optimización | `lbfgs` |
| `--seed` | Semilla aleatoria | `42` |

## MLflow Tracking

Todas las corridas se agrupan dentro del experimento:

```text
Iris_LogisticRegression_Experiments
```

Para cada ejecución se registran:

- Hiperparámetros del modelo.
- Semilla aleatoria.
- Tamaño del conjunto de prueba.
- Accuracy.
- F1-score macro.
- Matriz de confusión como artefacto.
- Pipeline entrenado de Scikit-learn.

## Experimentos realizados

Se realizaron cinco corridas principales modificando el hiperparámetro `C` y manteniendo constantes los demás parámetros.

| C | Accuracy | F1 Macro |
|---:|---:|---:|
| 0.01 | 0.800 | 0.792 |
| 0.1 | 0.867 | 0.865 |
| 1.0 | 0.933 | 0.933 |
| 10.0 | 1.000 | 1.000 |
| 100.0 | 1.000 | 1.000 |

La configuración seleccionada fue `C=10`, ya que obtuvo el mejor desempeño y alcanzó los mismos resultados que `C=100` utilizando un valor menos extremo.

## Reproducibilidad

La semilla aleatoria puede controlarse mediante el argumento:

```bash
--seed 42
```

El script fija la semilla utilizada y mantiene constante la división de entrenamiento y prueba cuando se utilizan los mismos argumentos.

Por lo tanto, dos ejecuciones realizadas con la misma configuración producen las mismas métricas.

## Estructura del repositorio

```text
.
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Autor

**Diego Elian Rodriguez Cantu**  
A00829925
