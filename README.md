# Session 406 ML Cloud Apps

Static GitHub Pages deployment for two Session 406 course tasks:

- MNIST browser app: canvas drawing, 28x28 crop/scale/center preprocessing, and a two-hidden-layer ReLU classifier. The deployed model is trained on the full MNIST dataset and reports 98.26% test accuracy.
- UFO browser app: duration/latitude/longitude inputs, an interactive map, country cluster profiles, and a browser-side classifier trained from the course UFO CSV. It predicts which country cluster a sighting most closely matches; it does not judge whether a sighting is real.

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
  --mnist-full-npz /path/to/mnist.npz \
  --ufo-csv /path/to/ufos.csv \
  --output-dir data
```
