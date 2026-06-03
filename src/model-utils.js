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

function sampleBilinear(source, width, height, x, y) {
  const clampedX = Math.max(0, Math.min(width - 1, x));
  const clampedY = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const dx = clampedX - x0;
  const dy = clampedY - y0;

  const top =
    source[y0 * width + x0] * (1 - dx) +
    source[y0 * width + x1] * dx;
  const bottom =
    source[y1 * width + x0] * (1 - dx) +
    source[y1 * width + x1] * dx;
  return top * (1 - dy) + bottom * dy;
}

export function normalizeDigitImage(source, sourceWidth, sourceHeight, options = {}) {
  const targetWidth = options.targetWidth ?? 28;
  const targetHeight = options.targetHeight ?? 28;
  const innerSize = options.innerSize ?? 20;
  const threshold = options.threshold ?? 0.08;
  const empty = Array.from({ length: targetWidth * targetHeight }, () => 0);

  let minX = sourceWidth;
  let minY = sourceHeight;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < sourceHeight; y += 1) {
    for (let x = 0; x < sourceWidth; x += 1) {
      const value = source[y * sourceWidth + x];
      if (value > threshold) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return empty;
  }

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const scale = Math.min(innerSize / cropWidth, innerSize / cropHeight);
  const scaledWidth = Math.max(1, Math.round(cropWidth * scale));
  const scaledHeight = Math.max(1, Math.round(cropHeight * scale));
  const offsetX = Math.floor((targetWidth - scaledWidth) / 2);
  const offsetY = Math.floor((targetHeight - scaledHeight) / 2);
  const centered = [...empty];

  for (let y = 0; y < scaledHeight; y += 1) {
    for (let x = 0; x < scaledWidth; x += 1) {
      const sourceX = minX + ((x + 0.5) / scaledWidth) * cropWidth - 0.5;
      const sourceY = minY + ((y + 0.5) / scaledHeight) * cropHeight - 0.5;
      centered[(offsetY + y) * targetWidth + offsetX + x] = sampleBilinear(
        source,
        sourceWidth,
        sourceHeight,
        sourceX,
        sourceY,
      );
    }
  }

  let mass = 0;
  let totalX = 0;
  let totalY = 0;
  centered.forEach((value, index) => {
    mass += value;
    totalX += (index % targetWidth) * value;
    totalY += Math.floor(index / targetWidth) * value;
  });
  if (!mass) {
    return empty;
  }

  const centerX = (targetWidth - 1) / 2;
  const centerY = (targetHeight - 1) / 2;
  const shiftX = Math.round(centerX - totalX / mass);
  const shiftY = Math.round(centerY - totalY / mass);
  const shifted = [...empty];
  centered.forEach((value, index) => {
    if (!value) return;
    const x = (index % targetWidth) + shiftX;
    const y = Math.floor(index / targetWidth) + shiftY;
    if (x >= 0 && x < targetWidth && y >= 0 && y < targetHeight) {
      shifted[y * targetWidth + x] = Math.max(shifted[y * targetWidth + x], value);
    }
  });

  return shifted;
}

export function projectGeoPoint(latitude, longitude, width, height) {
  const clampedLatitude = Math.max(-90, Math.min(90, latitude));
  const clampedLongitude = Math.max(-180, Math.min(180, longitude));
  return {
    x: ((clampedLongitude + 180) / 360) * width,
    y: ((90 - clampedLatitude) / 180) * height,
  };
}
