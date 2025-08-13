## Reto Técnico: Clasificación de Artículos Biomédicos con Machine Learning

Este repositorio contiene una solución completa y reproducible para clasificar artículos médicos en uno o varios grupos temáticos usando únicamente el `title` y el `abstract` como insumo. Los grupos objetivo son:

- Cardiovascular
- Neurological
- Hepatorenal
- Oncological

La solución incluye:
- Estructura modular en `src/` (código reutilizable y documentado)
- Scripts CLI en `scripts/` para entrenamiento, evaluación y predicción
- Reportes de métricas y artefactos del modelo versionados en `outputs/` y `reports/`
- Notebook de EDA en `notebooks/`
- Instrucciones paso a paso para ejecutar y validar

### 1) Requisitos

- Python 3.9+ (recomendado 3.10+)
- macOS/Linux/Windows

### 2) Configuración rápida del entorno

```bash
# 1) Crear y activar un entorno virtual (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 2) Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# (Opcional) Verificar NLTK stopwords la primera vez
python -c "import nltk; nltk.download('stopwords')"
```

### 3) Estructura del proyecto

```
technical-challenge-ml/
  data/                        # Pon aquí challenge_data.csv (no se versiona)
  outputs/                     # Modelos y resultados de entrenamiento
  reports/                     # Reportes (métricas, EDA, figuras)
  notebooks/
    01_eda.ipynb              # Exploración y visualización
  scripts/
    train.py                  # Entrenar y evaluar (genera modelo y reporte)
    predict.py                # Cargar modelo y predecir sobre un CSV
  src/
    __init__.py
    config.py
    data_loading.py
    preprocessing.py
    modeling.py
    evaluation.py
    utils.py
  requirements.txt
  README.md
```

### 4) Dataset

- Espera un CSV con columnas: `title`, `abstract`, `group`.
- `group` puede incluir uno o varios grupos; se aceptan separadores comunes (`,`, `|`, `;`, `/`).
- Ejemplo mínimo:

```csv
title,abstract,group
Study on cardiac function,...,Cardiovascular
Brain network activity,...,Neurological|Oncological
```

Coloca tu archivo en `data/challenge_data.csv`.

### 5) Entrenamiento y evaluación

```bash
python scripts/train.py \
  --data_path data/challenge_data.csv \
  --output_dir outputs \
  --model_type logistic \
  --test_size 0.2 \
  --random_state 42 \
  --cv 3
```

Esto:
- Entrena un pipeline `TF-IDF + OneVsRest + (LogisticRegression|LinearSVC)`
- Realiza búsqueda de hiperparámetros vía GridSearchCV
- Evalúa en un test set separado (F1 micro/macro y Subset Accuracy)
- Guarda el modelo, métricas y figuras bajo `outputs/<timestamp>_model/`

Parámetros principales:
- `--model_type`: `logistic` (por defecto) o `linearsvc`
- `--cv`: número de folds para validación interna en la búsqueda de hiperparámetros
- `--test_size`: tamaño del conjunto de prueba

### 6) Predicción sobre un CSV nuevo

```bash
python scripts/predict.py \
  --model_dir outputs/<timestamp>_model \
  --input_csv data/challenge_data.csv \
  --output_csv outputs/predictions.csv \
  --evaluate  # Si el CSV incluye la columna group, calcula métricas
```

Salida:
- `outputs/predictions.csv` con columnas originales y `predicted_groups`
- Si `--evaluate` y el CSV tiene `group`, se genera `metrics.json` y `report.txt`

### 7) Notebook de EDA

Abre `notebooks/01_eda.ipynb` y ejecuta las celdas. Genera:
- Distribución de etiquetas y co-ocurrencias
- Longitud de títulos/abstracts
- N-gramas más frecuentes
- Nubes de palabras (opcional)

### 8) Métricas y reportes

Se reportan:
- F1-score micro y macro
- Subset Accuracy (exact match)
- Reporte por clase (precision/recall/F1)
- Matriz de confusión por clase (one-vs-rest)

Archivos generados por `train.py`:
- `best_model.joblib`
- `label_binarizer.joblib`
- `metrics.json`
- `report.txt`
- Figuras en `figs/`

### 9) Buenas prácticas aplicadas

- PEP8, docstrings y tipado estático selectivo
- Semillas fijas para reproducibilidad
- Separación modular (código reusable en `src/`)
- Manejo robusto de formatos de `group`
- Evitar data leakage (se excluye `group` del pipeline de features)

### 10) Referencias rápidas de uso

```bash
# Entrenar con LogisticRegression
python scripts/train.py --data_path data/challenge_data.csv --model_type logistic

# Entrenar con LinearSVC
python scripts/train.py --data_path data/challenge_data.csv --model_type linearsvc

# Predecir (y evaluar si hay ground truth)
python scripts/predict.py --model_dir outputs/<timestamp>_model --input_csv data/challenge_data.csv --output_csv outputs/pred.csv --evaluate
```

### 11) Notas

- Si es la primera vez que usas NLTK, ejecuta `nltk.download('stopwords')`.
- Si tu dataset está en otro idioma, ajusta `STOPWORDS_LANGUAGE` en `src/config.py`.
- Para datasets muy desbalanceados, considera ajustar `class_weight` o probar `LinearSVC`.

