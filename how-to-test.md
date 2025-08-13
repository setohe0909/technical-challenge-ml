## Pasos rápidos para probar

- Usa tu archivo en: `/../examples/challenge_data.csv` (o muévelo a data/challenge_data.csv si prefieres).

## Prepara entorno

```bash
python3 -m venv /../.venv
source /../.venv/bin/activate
python -m pip install --upgrade pip
pip install -r /../requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Entrena y evalúa

```bash
python -m scripts.train --data_path examples/challenge_data.csv --output_dir outputs --model_type logistic --cv 3 --test_size 0.2 --sep ';'
```

```bash
python /../scripts/train.py \
  --data_path /../examples/challenge_data.csv \
  --output_dir /../outputs \
  --model_type logistic \
  --test_size 0.2 \
  --random_state 42 \
  --cv 3
```

## Verifica resultados

```bash
MODEL_DIR=$(ls -dt outputs/*_model | head -n 1)
cat "$MODEL_DIR/report.txt" | head -n 80
```

```bash
MODEL_DIR=$(ls -dt /.../outputs/*_model | head -n 1)
echo "$MODEL_DIR"
cat "$MODEL_DIR/report.txt" | head -n 60
```

## Genera predicciones y (opcional) re-evalúa

```bash
MODEL_DIR=$(ls -dt outputs/*_model | head -n 1)
python -m scripts.predict \
  --model_dir "$MODEL_DIR" \
  --input_csv examples/challenge_data.csv \
  --output_csv outputs/predictions.csv \
  --evaluate \
  --sep ';'
head -n 5 outputs/predictions.csv
```

## EDA en notebook

Abre notebooks/01_eda.ipynb y ejecuta.
Si no está en data/challenge_data.csv, cambia la variable DATA_PATH al inicio del notebook (por ejemplo a ../examples/challenge_data.csv).
Sugerencia si tu texto está en español
Cambia STOPWORDS_LANGUAGE = "spanish" en src/config.py y vuelve a entrenar.