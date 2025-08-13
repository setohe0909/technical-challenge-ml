## Pasos rápidos para probar

- Usa tu archivo en: `/Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/examples/challenge_data.csv` (o muévelo a data/challenge_data.csv si prefieres).

## Prepara entorno

```bash
python3 -m venv /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/.venv
source /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/.venv/bin/activate
python -m pip install --upgrade pip
pip install -r /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Entrena y evalúa

```bash
python /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/scripts/train.py \
  --data_path /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/examples/challenge_data.csv \
  --output_dir /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/outputs \
  --model_type logistic \
  --test_size 0.2 \
  --random_state 42 \
  --cv 3
```

## Verifica resultados

```bash
MODEL_DIR=$(ls -dt /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/outputs/*_model | head -n 1)
echo "$MODEL_DIR"
cat "$MODEL_DIR/report.txt" | head -n 60
```

## Genera predicciones y (opcional) re-evalúa

```bash
python /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/scripts/predict.py \
  --model_dir "$MODEL_DIR" \
  --input_csv /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/examples/challenge_data.csv \
  --output_csv /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/outputs/predictions.csv \
  --evaluate

head -n 5 /Users/sebastiantobon/Documents/repo/ml/technical-challenge-ml/outputs/predictions.csv
```

## EDA en notebook

Abre notebooks/01_eda.ipynb y ejecuta.
Si no está en data/challenge_data.csv, cambia la variable DATA_PATH al inicio del notebook (por ejemplo a ../examples/challenge_data.csv).
Sugerencia si tu texto está en español
Cambia STOPWORDS_LANGUAGE = "spanish" en src/config.py y vuelve a entrenar.