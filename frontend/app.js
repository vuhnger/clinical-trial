const API_BASE = "/api";

const clinicCountEl = document.getElementById("clinicCount");
const totalKmEl = document.getElementById("totalKm");
const elevationGainEl = document.getElementById("elevationGain");
const closeLoopToggle = document.getElementById("closeLoopToggle");
const selectAllBtn = document.getElementById("selectAllBtn");
const selectOsloBtn = document.getElementById("selectOsloBtn");
const generateBtn = document.getElementById("generateBtn");
const downloadBtn = document.getElementById("downloadBtn");
const clearBtn = document.getElementById("clearBtn");
const profileCanvas = document.getElementById("heightProfile");
const introModal = document.getElementById("introModal");
const introCloseBtn = document.getElementById("introCloseBtn");

const state = {
  clinics: [],
  markerById: new Map(),
  selectedIds: new Set(),
  routeLayer: null,
  lastRequestBody: null,
  activeStreamController: null,
  previewActive: false,
};
let hoverPopup = null;

const map = L.map("map", { zoomControl: true }).setView([59.9139, 10.7522], 11);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  maxZoom: 20,
  subdomains: "abcd",
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
}).addTo(map);

const logoIcon = L.icon({
  iconUrl: "/logo.svg",
  iconSize: [22, 22],
  iconAnchor: [11, 11],
  popupAnchor: [0, -10],
});

function setBusy(busy) {
  closeLoopToggle.disabled = busy;
  selectAllBtn.disabled = busy;
  selectOsloBtn.disabled = busy;
  generateBtn.disabled = busy;
  clearBtn.disabled = busy;
  downloadBtn.disabled = busy || !state.lastRequestBody;
  generateBtn.textContent = busy ? "Beregner..." : "Generer rute";
}

function showIntroModal() {
  if (!introModal) return;
  introModal.classList.remove("is-hidden");
}

function hideIntroModal() {
  if (!introModal) return;
  introModal.classList.add("is-hidden");
  localStorage.setItem("routeIntroSeen", "true");
}

function updateHeader(distanceKm = null, elevationGainM = null) {
  clinicCountEl.textContent = String(state.selectedIds.size);
  if (distanceKm == null) {
    totalKmEl.textContent = "0.0";
  } else {
    totalKmEl.textContent = Number(distanceKm).toFixed(1);
  }
  if (elevationGainM == null) {
    elevationGainEl.textContent = "0 m";
  } else {
    elevationGainEl.textContent = `${Math.max(0, Math.round(Number(elevationGainM)))} m`;
  }
}

function selectedMarkerStyle() {
  return {
    radius: 8,
    color: "#FFFFFF",
    weight: 2,
    fillColor: "#75D0C5",
    fillOpacity: 1,
  };
}

function unselectedMarkerStyle() {
  return {
    radius: 7,
    color: "#FFFFFF",
    weight: 1.5,
    fillColor: "#2E4F4E",
    fillOpacity: 0.9,
  };
}

function toggleClinic(clinicId) {
  if (state.selectedIds.has(clinicId)) {
    state.selectedIds.delete(clinicId);
  } else {
    state.selectedIds.add(clinicId);
  }
  const marker = state.markerById.get(clinicId);
  if (marker) {
    marker.setStyle(state.selectedIds.has(clinicId) ? selectedMarkerStyle() : unselectedMarkerStyle());
  }
  updateHeader(null);
  state.lastRequestBody = null;
  downloadBtn.disabled = true;
}

function selectClinics(filterFn) {
  if (state.activeStreamController) {
    state.activeStreamController.abort();
    state.activeStreamController = null;
  }
  state.selectedIds.clear();
  for (const clinic of state.clinics) {
    if (!filterFn || filterFn(clinic)) {
      state.selectedIds.add(clinic.id);
    }
  }
  for (const clinic of state.clinics) {
    const marker = state.markerById.get(clinic.id);
    if (marker) {
      marker.setStyle(state.selectedIds.has(clinic.id) ? selectedMarkerStyle() : unselectedMarkerStyle());
    }
  }
  if (state.routeLayer) {
    map.removeLayer(state.routeLayer);
    state.routeLayer = null;
  }
  state.previewActive = false;
  state.lastRequestBody = null;
  downloadBtn.disabled = true;
  updateHeader(null, null);
  drawHeightProfile([], 0);
}

function closeHoverPopup() {
  if (hoverPopup) {
    map.closePopup(hoverPopup);
    hoverPopup = null;
  }
}

async function loadClinics() {
  const response = await fetch(`${API_BASE}/clinics`);
  if (!response.ok) {
    throw new Error("Kunne ikke hente klinikker.");
  }
  const payload = await response.json();
  state.clinics = payload.clinics || [];

  const bounds = [];
  for (const clinic of state.clinics) {
    const marker = L.circleMarker([clinic.lat, clinic.lon], unselectedMarkerStyle()).addTo(map);
    const infoHtml = `<strong>${clinic.navn}</strong><br/>${clinic.gateadresse}, ${clinic.postnummer} ${clinic.kommune}`;
    marker.on("mouseover", (event) => {
      closeHoverPopup();
      hoverPopup = L.popup({
        closeButton: false,
        autoClose: false,
        closeOnClick: false,
        offset: [0, -8],
      })
        .setLatLng(event.latlng)
        .setContent(infoHtml)
        .openOn(map);
    });
    marker.on("mouseout", () => {
      closeHoverPopup();
    });
    marker.on("click", () => {
      closeHoverPopup();
      toggleClinic(clinic.id);
    });

    // Small logo overlay on top of the circle marker for visual branding.
    const logo = L.marker([clinic.lat, clinic.lon], { icon: logoIcon, interactive: false }).addTo(map);

    state.markerById.set(clinic.id, marker);
    bounds.push([clinic.lat, clinic.lon]);
  }

  if (bounds.length > 0) {
    map.fitBounds(bounds, { padding: [28, 28] });
  }

  selectClinics(() => true);
}

function clearRoute() {
  if (state.activeStreamController) {
    state.activeStreamController.abort();
    state.activeStreamController = null;
  }
  closeHoverPopup();
  if (state.routeLayer) {
    map.removeLayer(state.routeLayer);
    state.routeLayer = null;
  }
  for (const id of state.selectedIds) {
    const marker = state.markerById.get(id);
    if (marker) marker.setStyle(unselectedMarkerStyle());
  }
  state.selectedIds.clear();
  state.lastRequestBody = null;
  state.previewActive = false;
  downloadBtn.disabled = true;
  updateHeader(null, null);
  drawHeightProfile([], 0);
}

function routeStyle(isPreview) {
  return {
    color: "#75D0C5",
    weight: isPreview ? 3 : 4,
    opacity: isPreview ? 0.75 : 0.95,
    dashArray: isPreview ? "8 8" : null,
    lineJoin: "round",
  };
}

function drawRoute(routePoints, isPreview = false) {
  const latlngs = routePoints.map((p) => [p.lat, p.lon]);
  if (latlngs.length === 0) {
    return;
  }
  if (!state.routeLayer) {
    state.routeLayer = L.polyline(latlngs, routeStyle(isPreview)).addTo(map);
  } else {
    state.routeLayer.setLatLngs(latlngs);
    // Skip redundant setStyle during consecutive preview updates — style is unchanged.
    if (!isPreview || !state.previewActive) {
      state.routeLayer.setStyle(routeStyle(isPreview));
    }
  }
  if (isPreview) {
    if (!state.previewActive) {
      map.fitBounds(state.routeLayer.getBounds(), { padding: [28, 28] });
      state.previewActive = true;
    }
  } else {
    state.previewActive = false;
    map.fitBounds(state.routeLayer.getBounds(), { padding: [28, 28] });
  }
}

async function yieldToBrowser() {
  await new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function calculateElevationGain(profile, minStepM = 2) {
  if (!profile || profile.length < 2) return 0;
  let gain = 0;
  for (let i = 1; i < profile.length; i += 1) {
    const delta = Number(profile[i].elev) - Number(profile[i - 1].elev);
    if (Number.isFinite(delta) && delta >= minStepM) gain += delta;
  }
  return Math.round(gain);
}

function drawHeightProfile(profile, elevationGainM = null) {
  const ctx = profileCanvas.getContext("2d");
  const w = profileCanvas.width;
  const h = profileCanvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(8, h - 20);
  ctx.lineTo(w - 8, h - 20);
  ctx.stroke();

  if (!profile || profile.length < 2) {
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "12px Segoe UI, Arial";
    ctx.fillText("Høydeprofil • 0 m", 10, 16);
    return;
  }

  const kms = profile.map((p) => p.km);
  const elevs = profile.map((p) => p.elev);
  const kmMin = Math.min(...kms);
  const kmMax = Math.max(...kms);
  const elevMin = Math.min(...elevs);
  const elevMax = Math.max(...elevs);
  const elevSpan = Math.max(elevMax - elevMin, 1);
  const kmSpan = Math.max(kmMax - kmMin, 0.001);

  ctx.strokeStyle = "#75D0C5";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  profile.forEach((point, idx) => {
    const x = 10 + ((point.km - kmMin) / kmSpan) * (w - 20);
    const y = 20 + (1 - (point.elev - elevMin) / elevSpan) * (h - 40);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#FFFFFF";
  ctx.font = "12px Segoe UI, Arial";
  const gain = elevationGainM == null ? calculateElevationGain(profile) : Math.round(Number(elevationGainM));
  const shownGain = Number.isFinite(gain) ? Math.max(0, gain) : 0;
  ctx.fillText(`Høydeprofil • ${shownGain} m`, 10, 16);
}

function routeRequestBody() {
  return {
    clinic_ids: Array.from(state.selectedIds),
    random_starts: 900,
    two_opt_rounds: 140,
    close_loop: Boolean(closeLoopToggle.checked),
  };
}

function parseSseBlock(block) {
  const lines = block.split(/\r?\n/);
  let event = "message";
  const dataParts = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataParts.push(line.slice(5).trim());
    }
  }
  const dataText = dataParts.join("\n");
  let data = {};
  if (dataText) {
    try {
      data = JSON.parse(dataText);
    } catch {
      data = { raw: dataText };
    }
  }
  return { event, data };
}

async function consumeSseResponse(response, handlers) {
  if (!response.body) {
    throw new Error("Mangler stream-body fra API.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let previewEventsSinceYield = 0;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        // Flush any trailing partial SSE block if the stream ended without \n\n.
        const trailing = buffer.replace(/\r/g, "").trim();
        if (trailing.length > 0) {
          const parsed = parseSseBlock(trailing);
          const fn = handlers[parsed.event];
          if (fn) {
            const maybePromise = fn(parsed.data);
            if (maybePromise && typeof maybePromise.then === "function") {
              await maybePromise;
            }
          }
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r/g, "");
      let splitIdx = buffer.indexOf("\n\n");
      while (splitIdx >= 0) {
        const block = buffer.slice(0, splitIdx).trim();
        buffer = buffer.slice(splitIdx + 2);
        if (block.length > 0) {
          const parsed = parseSseBlock(block);
          const fn = handlers[parsed.event];
          if (fn) {
            const maybePromise = fn(parsed.data);
            if (maybePromise && typeof maybePromise.then === "function") {
              await maybePromise;
            }
          }
          if (parsed.event === "preview") {
            previewEventsSinceYield += 1;
            if (previewEventsSinceYield >= 2) {
              previewEventsSinceYield = 0;
              await yieldToBrowser();
            }
          }
        }
        splitIdx = buffer.indexOf("\n\n");
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // Swallow cancel errors to avoid masking the original error.
    }
  }
}

async function generateRoute() {
  if (state.selectedIds.size < 2) {
    alert("Velg minst to klinikker i kartet.");
    return;
  }

  if (state.activeStreamController) {
    state.activeStreamController.abort();
  }
  const controller = new AbortController();
  state.activeStreamController = controller;
  state.previewActive = false;

  setBusy(true);
  try {
    const body = routeRequestBody();
    const response = await fetch(`${API_BASE}/route/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Kunne ikke beregne rute.");
    }

    let finalPayload = null;
    let streamError = null;
    await consumeSseResponse(response, {
      status: () => {},
      preview: (data) => {
        if (data && Array.isArray(data.route) && data.route.length > 1) {
          drawRoute(data.route, true);
        }
      },
      result: (data) => {
        finalPayload = data;
      },
      error: (data) => {
        streamError = data && data.detail ? data.detail : "Ukjent stream-feil";
      },
    });

    if (streamError) {
      throw new Error(streamError);
    }
    if (!finalPayload) {
      throw new Error("Mottok ikke sluttresultat fra stream.");
    }

    drawRoute(finalPayload.route || [], false);
    const elevationGain =
      finalPayload.elevation_gain_m ?? calculateElevationGain(finalPayload.profile || []);
    drawHeightProfile(finalPayload.profile || [], elevationGain);
    updateHeader(finalPayload.distance_km, elevationGain);
    state.lastRequestBody = body;
    downloadBtn.disabled = false;
  } catch (error) {
    if (error.name !== "AbortError") {
      alert(`Feil: ${error.message || error}`);
    }
  } finally {
    if (state.activeStreamController === controller) {
      state.activeStreamController = null;
      setBusy(false);
    }
  }
}

async function downloadGpx() {
  if (!state.lastRequestBody) return;

  downloadBtn.disabled = true;
  try {
    const response = await fetch(`${API_BASE}/route/gpx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.lastRequestBody),
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Kunne ikke laste ned GPX.");
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename=\"([^\"]+)\"/);
    const filename = match ? match[1] : "drdropin-loperute.gpx";

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(`Feil: ${error.message || error}`);
  } finally {
    downloadBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", generateRoute);
selectAllBtn.addEventListener("click", () => selectClinics(() => true));
selectOsloBtn.addEventListener("click", () =>
  selectClinics((clinic) => String(clinic.omrade || "").toLowerCase() === "oslo")
);
downloadBtn.addEventListener("click", downloadGpx);
clearBtn.addEventListener("click", clearRoute);

if (introCloseBtn) {
  introCloseBtn.addEventListener("click", hideIntroModal);
}
if (introModal) {
  introModal.addEventListener("click", (event) => {
    if (event.target === introModal) {
      hideIntroModal();
    }
  });
}

if (!localStorage.getItem("routeIntroSeen")) {
  showIntroModal();
}

drawHeightProfile([], 0);
updateHeader(null, null);
loadClinics().catch((err) => {
  alert(err.message || "Kunne ikke laste kartdata.");
});
