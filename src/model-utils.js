export function standardizeFeatures(features, scaler) {
  return features.map((value, index) => {
    const scale = scaler.scale[index] || 1;
    return (value - scaler.mean[index]) / scale;
  });
}

export function predictLinearSoftmax(model, rawFeatures) {
  const features = standardizeFeatures(rawFeatures, model.scaler);
  const logits = model.coefficients.map((weights, classIndex) => {
    const weighted = weights.reduce(
      (total, weight, featureIndex) => total + weight * features[featureIndex],
      0,
    );
    return weighted + model.intercepts[classIndex];
  });
  const maxLogit = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - maxLogit));
  const total = exps.reduce((sum, value) => sum + value, 0);

  return logits
    .map((_, index) => ({
      label: model.labels[index],
      probability: exps[index] / total,
    }))
    .sort((a, b) => b.probability - a.probability);
}

export function predictDenseNetwork(model, rawFeatures) {
  let activations = standardizeFeatures(rawFeatures, model.scaler);

  for (const layer of model.layers) {
    const next = layer.weights.map((weights, outputIndex) => {
      const weighted = weights.reduce(
        (total, weight, inputIndex) => total + weight * activations[inputIndex],
        layer.biases[outputIndex],
      );
      return layer.activation === "relu" ? Math.max(0, weighted) : weighted;
    });
    activations = next;
  }

  const maxLogit = Math.max(...activations);
  const exps = activations.map((logit) => Math.exp(logit - maxLogit));
  const total = exps.reduce((sum, value) => sum + value, 0);

  return activations
    .map((_, index) => ({
      label: model.labels[index],
      probability: exps[index] / total,
    }))
    .sort((a, b) => b.probability - a.probability);
}

export function topK(entries, count) {
  return [...entries]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, count);
}

export function downsampleGrid(source, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const result = [];
  const cellWidth = sourceWidth / targetWidth;
  const cellHeight = sourceHeight / targetHeight;

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const startX = Math.floor(targetX * cellWidth);
      const endX = Math.floor((targetX + 1) * cellWidth);
      const startY = Math.floor(targetY * cellHeight);
      const endY = Math.floor((targetY + 1) * cellHeight);
      let total = 0;
      let count = 0;

      for (let y = startY; y < endY; y += 1) {
        for (let x = startX; x < endX; x += 1) {
          total += source[y * sourceWidth + x];
          count += 1;
        }
      }

      result.push(count ? total / count : 0);
    }
  }

  return result;
}
