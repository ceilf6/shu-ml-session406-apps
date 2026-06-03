import {
  downsampleGrid,
  predictDenseNetwork,
  predictLinearSoftmax,
  topK,
} from "./model-utils.js";

const state = {
  digitModel: null,
  ufoModel: null,
  activeView: "mnist",
};

const formatPercent = new Intl.NumberFormat("en", {
  maximumFractionDigits: 1,
  style: "percent",
});

function byId(id) {
  return document.getElementById(id);
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

function setActiveView(view) {
  state.activeView = view;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  byId("mnist-view").classList.toggle("active", view === "mnist");
  byId("ufo-view").classList.toggle("active", view === "ufo");
  if (location.hash !== `#${view}`) {
    history.replaceState(null, "", `#${view}`);
  }
}

function renderBars(container, predictions, count = 5) {
  const rows = topK(predictions, count)
    .map((entry) => {
      const width = `${Math.max(entry.probability * 100, 1).toFixed(2)}%`;
      return `
        <div class="bar-row">
          <div class="bar-top">
            <strong>${entry.label}</strong>
            <span>${formatPercent.format(entry.probability)}</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width: ${width}"></div></div>
        </div>
      `;
    })
    .join("");
  container.innerHTML = rows;
}

function setupTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => setActiveView(button.dataset.view));
  });
  window.addEventListener("hashchange", () => {
    setActiveView(location.hash === "#ufo" ? "ufo" : "mnist");
  });
  setActiveView(location.hash === "#ufo" ? "ufo" : "mnist");
}

function setupDigitCanvas() {
  const canvas = byId("digit-canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const brush = byId("brush-size");
  let isDrawing = false;

  function clearCanvas() {
    context.fillStyle = "#050806";
    context.fillRect(0, 0, canvas.width, canvas.height);
    byId("digit-label").textContent = "Waiting for input";
    byId("digit-bars").innerHTML = "";
    byId("digit-status").classList.remove("ready");
  }

  function positionFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const source = event.touches ? event.touches[0] : event;
    return {
      x: ((source.clientX - rect.left) / rect.width) * canvas.width,
      y: ((source.clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  function draw(event) {
    if (!isDrawing) return;
    event.preventDefault();
    const { x, y } = positionFromEvent(event);
    context.lineTo(x, y);
    context.strokeStyle = "#f8f7f1";
    context.lineWidth = Number(brush.value);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.stroke();
  }

  function begin(event) {
    isDrawing = true;
    const { x, y } = positionFromEvent(event);
    context.beginPath();
    context.moveTo(x, y);
    draw(event);
  }

  function end() {
    isDrawing = false;
  }

  canvas.addEventListener("pointerdown", begin);
  canvas.addEventListener("pointermove", draw);
  window.addEventListener("pointerup", end);
  canvas.addEventListener("touchstart", begin, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  window.addEventListener("touchend", end);
  byId("clear-digit").addEventListener("click", clearCanvas);
  byId("predict-digit").addEventListener("click", predictDigit);
  clearCanvas();
}

function extractDigitFeatures() {
  const canvas = byId("digit-canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const image = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const luminance = [];
  for (let index = 0; index < image.length; index += 4) {
    luminance.push((image[index] + image[index + 1] + image[index + 2]) / (3 * 255));
  }
  return downsampleGrid(luminance, canvas.width, canvas.height, 28, 28);
}

function predictDigit() {
  if (!state.digitModel) return;
  const features = extractDigitFeatures();
  const predictions = predictDenseNetwork(state.digitModel, features);
  byId("digit-label").textContent = `Digit ${predictions[0].label}`;
  byId("digit-status").classList.add("ready");
  renderBars(byId("digit-bars"), predictions, 5);
}

function drawUfoMap(latitude, longitude) {
  const canvas = byId("ufo-map");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0d1814";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(248, 247, 241, 0.16)";
  context.lineWidth = 1;
  for (let lon = -180; lon <= 180; lon += 30) {
    const x = ((lon + 180) / 360) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = ((90 - lat) / 180) * height;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  const x = ((longitude + 180) / 360) * width;
  const y = ((90 - latitude) / 180) * height;
  context.strokeStyle = "#47c76f";
  context.fillStyle = "#e46f2e";
  context.beginPath();
  context.arc(x, y, 8, 0, Math.PI * 2);
  context.fill();
  context.beginPath();
  context.moveTo(x - 18, y);
  context.lineTo(x + 18, y);
  context.moveTo(x, y - 18);
  context.lineTo(x, y + 18);
  context.stroke();
}

function setupUfoForm() {
  const form = byId("ufo-form");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    predictUfo();
  });
  form.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", drawCurrentUfoMap);
  });
  drawCurrentUfoMap();
}

function drawCurrentUfoMap() {
  drawUfoMap(Number(byId("ufo-latitude").value), Number(byId("ufo-longitude").value));
}

function predictUfo() {
  if (!state.ufoModel) return;
  const features = [
    Number(byId("ufo-seconds").value),
    Number(byId("ufo-latitude").value),
    Number(byId("ufo-longitude").value),
  ];
  const predictions = predictLinearSoftmax(state.ufoModel, features);
  byId("ufo-label").textContent = predictions[0].label;
  byId("ufo-status").classList.add("ready");
  renderBars(byId("ufo-bars"), predictions, 5);
  drawCurrentUfoMap();
}

async function init() {
  setupTabs();
  setupDigitCanvas();
  setupUfoForm();

  const [digitModel, ufoModel] = await Promise.all([
    loadJson("./data/digit-model.json"),
    loadJson("./data/ufo-model.json"),
  ]);
  state.digitModel = digitModel;
  state.ufoModel = ufoModel;
  byId("digit-accuracy").textContent = `Test ${formatPercent.format(digitModel.metadata.testAccuracy)}`;
  byId("ufo-accuracy").textContent = `Test ${formatPercent.format(ufoModel.metadata.testAccuracy)}`;
  predictUfo();
}

init().then(() => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
});
