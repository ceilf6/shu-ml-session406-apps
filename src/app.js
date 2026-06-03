import {
  normalizeDigitImage,
  predictDenseNetwork,
  predictLinearSoftmax,
  projectGeoPoint,
  topK,
} from "./model-utils.js";

const state = {
  digitModel: null,
  ufoModel: null,
  ufoMap: null,
  ufoInputMarker: null,
  activeView: "mnist",
};

const UFO_PRESETS = {
  sf: { seconds: 20, latitude: 37.7749, longitude: -122.4194 },
  toronto: { seconds: 18, latitude: 43.6532, longitude: -79.3832 },
  london: { seconds: 15, latitude: 51.5074, longitude: -0.1278 },
  sydney: { seconds: 22, latitude: -33.8688, longitude: 151.2093 },
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

function formatInteger(value) {
  return new Intl.NumberFormat("en").format(value);
}

function setActiveView(view) {
  state.activeView = view;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  byId("mnist-view").classList.toggle("active", view === "mnist");
  byId("ufo-view").classList.toggle("active", view === "ufo");
  if (view === "ufo" && state.ufoMap) {
    window.requestAnimationFrame(() => state.ufoMap.invalidateSize());
  }
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
    renderDigitPreview(Array.from({ length: 28 * 28 }, () => 0));
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
  return normalizeDigitImage(luminance, canvas.width, canvas.height);
}

function renderDigitPreview(features) {
  const canvas = byId("digit-preview");
  const context = canvas.getContext("2d");
  const image = context.createImageData(28, 28);
  features.forEach((value, index) => {
    const channel = Math.round(Math.max(0, Math.min(1, value)) * 255);
    image.data[index * 4] = channel;
    image.data[index * 4 + 1] = channel;
    image.data[index * 4 + 2] = channel;
    image.data[index * 4 + 3] = 255;
  });
  context.putImageData(image, 0, 0);
}

function predictDigit() {
  if (!state.digitModel) return;
  const features = extractDigitFeatures();
  renderDigitPreview(features);
  const predictions = predictDenseNetwork(state.digitModel, features);
  byId("digit-label").textContent = `Digit ${predictions[0].label}`;
  byId("digit-status").classList.add("ready");
  renderBars(byId("digit-bars"), predictions, 5);
}

function renderFallbackUfoMap(latitude, longitude) {
  const container = byId("ufo-map");
  let canvas = container.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.width = 680;
    canvas.height = 360;
    container.replaceChildren(canvas);
  }
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0d1814";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(248, 247, 241, 0.16)";
  context.lineWidth = 1;
  for (let lon = -180; lon <= 180; lon += 30) {
    const { x } = projectGeoPoint(0, lon, width, height);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    const { y } = projectGeoPoint(lat, 0, width, height);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  state.ufoModel?.metadata.countryProfiles.forEach((profile) => {
    const point = projectGeoPoint(profile.centroid.latitude, profile.centroid.longitude, width, height);
    context.fillStyle = "rgba(71, 199, 111, 0.52)";
    context.beginPath();
    context.arc(point.x, point.y, 5, 0, Math.PI * 2);
    context.fill();
  });

  const { x, y } = projectGeoPoint(latitude, longitude, width, height);
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

function setupUfoMap() {
  const container = byId("ufo-map");
  if (window.L) {
    state.ufoMap = window.L.map(container, {
      attributionControl: false,
      scrollWheelZoom: false,
      worldCopyJump: true,
    }).setView([20, 0], 2);
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 8,
    }).addTo(state.ufoMap);
    state.ufoModel.metadata.countryProfiles.forEach((profile) => {
      window.L.circleMarker([profile.centroid.latitude, profile.centroid.longitude], {
        radius: Math.max(5, Math.min(16, Math.sqrt(profile.count) / 10)),
        color: "#126e82",
        fillColor: "#47c76f",
        fillOpacity: 0.42,
        weight: 2,
      })
        .bindTooltip(`${profile.label}: ${formatInteger(profile.count)} reports`)
        .addTo(state.ufoMap);
    });
    state.ufoInputMarker = window.L.circleMarker([37.7749, -122.4194], {
      radius: 9,
      color: "#121411",
      fillColor: "#e46f2e",
      fillOpacity: 0.95,
      weight: 3,
    }).addTo(state.ufoMap);
  }
  renderUfoProfiles();
  drawCurrentUfoMap();
}

function drawUfoMap(latitude, longitude) {
  if (state.ufoMap && state.ufoInputMarker) {
    state.ufoInputMarker.setLatLng([latitude, longitude]);
    state.ufoMap.setView([latitude, longitude], Math.abs(latitude) > 55 ? 2 : 3, { animate: false });
    return;
  }
  renderFallbackUfoMap(latitude, longitude);
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
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = UFO_PRESETS[button.dataset.preset];
      byId("ufo-seconds").value = preset.seconds;
      byId("ufo-latitude").value = preset.latitude;
      byId("ufo-longitude").value = preset.longitude;
      predictUfo();
    });
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
  renderUfoProfiles(predictions[0].label);
  renderUfoExplanation(predictions[0]);
  drawCurrentUfoMap();
}

function renderUfoExplanation(topPrediction) {
  const profile = state.ufoModel.metadata.countryProfiles.find((entry) => entry.label === topPrediction.label);
  if (!profile) return;
  byId("ufo-explanation").textContent = `${profile.code} cluster: ${formatInteger(
    profile.count,
  )} reports, ${formatPercent.format(profile.share)} of the filtered dataset, median duration ${
    profile.centroid.seconds
  }s.`;
}

function renderUfoProfiles(activeLabel = "") {
  if (!state.ufoModel) return;
  byId("ufo-profiles").innerHTML = state.ufoModel.metadata.countryProfiles
    .map(
      (profile) => `
        <div class="profile-item ${profile.label === activeLabel ? "active" : ""}">
          <span>${profile.code}</span>
          <strong>${profile.label}</strong>
          <small>${formatInteger(profile.count)} reports · ${profile.centroid.latitude.toFixed(1)}, ${profile.centroid.longitude.toFixed(1)}</small>
        </div>
      `,
    )
    .join("");
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
  setupUfoMap();
  predictUfo();
}

init().then(() => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  document.body.classList.add("ready");
});
