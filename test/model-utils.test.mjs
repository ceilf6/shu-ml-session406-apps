import assert from "node:assert/strict";
import { test } from "node:test";

import {
  downsampleGrid,
  normalizeDigitImage,
  predictDenseNetwork,
  predictLinearSoftmax,
  projectGeoPoint,
  standardizeFeatures,
  topK,
} from "../src/model-utils.js";

test("standardizeFeatures applies model mean and scale per feature", () => {
  const result = standardizeFeatures([4, 10], {
    mean: [2, 2],
    scale: [2, 4],
  });

  assert.deepEqual(result, [1, 2]);
});

test("predictLinearSoftmax returns sorted label probabilities", () => {
  const model = {
    labels: ["cold", "warm"],
    scaler: { mean: [0, 0], scale: [1, 1] },
    coefficients: [
      [-1, -1],
      [1, 1],
    ],
    intercepts: [0, 0],
  };

  const prediction = predictLinearSoftmax(model, [2, 1]);

  assert.equal(prediction[0].label, "warm");
  assert.ok(prediction[0].probability > 0.99);
  assert.equal(prediction.length, 2);
});

test("predictDenseNetwork applies relu hidden layers and sorted softmax output", () => {
  const model = {
    labels: ["zero", "one"],
    scaler: { mean: [0, 0], scale: [1, 1] },
    layers: [
      {
        weights: [
          [1, 0],
          [0, -1],
        ],
        biases: [0, 0],
        activation: "relu",
      },
      {
        weights: [
          [1, -1],
          [-1, 1],
        ],
        biases: [0, 0],
        activation: "linear",
      },
    ],
  };

  const prediction = predictDenseNetwork(model, [2, 1]);

  assert.equal(prediction[0].label, "zero");
  assert.ok(prediction[0].probability > 0.98);
});

test("topK returns the requested number of highest probability entries", () => {
  const entries = [
    { label: "a", probability: 0.1 },
    { label: "b", probability: 0.7 },
    { label: "c", probability: 0.2 },
  ];

  assert.deepEqual(topK(entries, 2), [
    { label: "b", probability: 0.7 },
    { label: "c", probability: 0.2 },
  ]);
});

test("downsampleGrid averages source cells into a smaller grid", () => {
  const source = [
    1, 1, 0, 0,
    1, 1, 0, 0,
    0, 0, 0.5, 0.5,
    0, 0, 0.5, 0.5,
  ];

  const result = downsampleGrid(source, 4, 4, 2, 2);

  assert.deepEqual(result, [1, 0, 0, 0.5]);
});

test("normalizeDigitImage centers an off-axis handwritten mark", () => {
  const source = Array.from({ length: 8 * 8 }, () => 0);
  source[1 * 8 + 1] = 1;
  source[1 * 8 + 2] = 1;
  source[2 * 8 + 1] = 1;
  source[2 * 8 + 2] = 1;

  const result = normalizeDigitImage(source, 8, 8, {
    targetWidth: 8,
    targetHeight: 8,
    innerSize: 4,
    threshold: 0.05,
  });
  const activeXs = result
    .map((value, index) => (value > 0.1 ? index % 8 : -1))
    .filter((value) => value >= 0);
  const activeYs = result
    .map((value, index) => (value > 0.1 ? Math.floor(index / 8) : -1))
    .filter((value) => value >= 0);

  assert.equal(result.length, 64);
  assert.ok(Math.min(...activeXs) > 0);
  assert.ok(Math.max(...activeXs) < 7);
  assert.ok(Math.min(...activeYs) > 0);
  assert.ok(Math.max(...activeYs) < 7);
});

test("normalizeDigitImage returns an empty target for a blank canvas", () => {
  const result = normalizeDigitImage(Array.from({ length: 16 }, () => 0), 4, 4, {
    targetWidth: 4,
    targetHeight: 4,
  });

  assert.deepEqual(result, Array.from({ length: 16 }, () => 0));
});

test("projectGeoPoint maps latitude and longitude into viewport coordinates", () => {
  assert.deepEqual(projectGeoPoint(0, 0, 360, 180), { x: 180, y: 90 });
  assert.deepEqual(projectGeoPoint(90, -180, 360, 180), { x: 0, y: 0 });
  assert.deepEqual(projectGeoPoint(-90, 180, 360, 180), { x: 360, y: 180 });
});
