# Session 406 ML Cloud Apps

Static GitHub Pages deployment for two Session 406 course tasks:

- MNIST browser app: canvas drawing plus a one-hidden-layer ReLU classifier trained from the course MNIST CSV.
- UFO browser app: duration/latitude/longitude inputs plus a country classifier trained from the course UFO CSV.

The apps run entirely in the browser, so GitHub Pages is enough. No Streamlit or FastAPI server is required.

## Local Development

```bash
npm test
python3 -m http.server 4173
```

## Regenerate Models

```bash
python scripts/train_models.py \
  --mnist-train-csv /path/to/mnist_train.csv \
  --mnist-test-csv /path/to/mnist_test.csv \
  --ufo-csv /path/to/ufos.csv \
  --output-dir data
```
