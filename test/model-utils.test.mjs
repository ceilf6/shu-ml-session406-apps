import assert from "node:assert/strict";
import { test } from "node:test";

import {
  downsampleGrid,
  predictDenseNetwork,
  predictLinearSoftmax,
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
