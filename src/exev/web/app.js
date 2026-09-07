const panels = document.querySelector("#panels");
const template = document.querySelector("#panel-template");
const summary = document.querySelector("#summary");
let data;
let index = 0;
let playing = false;
let timer;
let routerOptions = [];
let disasterOptions = [];
let osmDocument = null;
let osmPreviewNodes = [];

const colors = {
  road: "#4b4b4b", agent: "#ffffff", waiting: "#a9a9a9",
  hazard: "#e2231a", blocked: "#ff352c", shelter: "#087ea4",
};

function setup(run, panelIndex) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".panel");
  const select = article.querySelector(".router-select");
  for (const name of routerOptions) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === run.algorithm;
    select.append(option);
  }
  select.onchange = () => switchRouter(panelIndex, select.value);
  panels.append(node);
}

function rebuild() {
  panels.replaceChildren();
  data.runs.forEach(setup);
  document.querySelector("#timeline").max = maxIndex();
  index = Math.min(index, maxIndex());
  draw();
}

async function switchRouter(panelIndex, name) {
  const names = data.runs.map((run) => run.algorithm);
  const previous = names[panelIndex];
  const other = panelIndex === 0 ? 1 : 0;
  names[panelIndex] = name;
  if (names.length === 2 && names[other] === name) names[other] = previous;
  await loadScenario(names, document.querySelector("#hazard").value);
}

async function loadScenario(names, disaster) {
  if (data.map_type === "openstreetmap" && osmDocument) {
    await loadOSM(names);
    return;
  }
  panels.classList.add("loading");
  summary.hidden = true;
  document.querySelector("#replay").hidden = true;
  setPlaying(false);
  try {
    const query = `routers=${encodeURIComponent(names.join(","))}&disaster=${encodeURIComponent(disaster)}`;
    const response = await fetch(`/api/simulation?${query}`);
    if (!response.ok) throw new Error("scenario request failed");
    data = await response.json();
    index = 0;
    updateScenarioLabel();
    rebuild();
  } catch (error) {
    rebuild();
  } finally {
    panels.classList.remove("loading");
  }
}

function previewOSM(documentText) {
  const xml = new DOMParser().parseFromString(documentText, "application/xml");
  if (xml.querySelector("parsererror")) throw new Error("Invalid OSM XML file.");
  const coordinates = Object.fromEntries([...xml.querySelectorAll("node")].map((node) => [
    node.getAttribute("id"),
    {id: node.getAttribute("id"), x: Number(node.getAttribute("lon")), y: -Number(node.getAttribute("lat"))},
  ]));
  const ways = [...xml.querySelectorAll("way")].filter((way) =>
    [...way.querySelectorAll("tag")].some((tag) => tag.getAttribute("k") === "highway"),
  ).map((way) => [...way.querySelectorAll("nd")].map((node) => node.getAttribute("ref")));
  const used = new Set(ways.flat());
  osmPreviewNodes = Object.values(coordinates).filter((node) => used.has(node.id));
  const canvas = document.querySelector("#osm-preview");
  canvas.hidden = false;
  const width = canvas.clientWidth, height = canvas.clientHeight, dpr = devicePixelRatio || 1;
  canvas.width = width * dpr; canvas.height = height * dpr;
  const context = canvas.getContext("2d"); context.scale(dpr, dpr);
  const xs = osmPreviewNodes.map((node) => node.x), ys = osmPreviewNodes.map((node) => node.y), pad = 18;
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (x) => pad + (x - minX) / (maxX - minX || 1) * (width - pad * 2);
  const sy = (y) => pad + (y - minY) / (maxY - minY || 1) * (height - pad * 2);
  context.strokeStyle = "#8c8c8c"; context.lineWidth = 2;
  for (const way of ways) {
    context.beginPath();
    way.forEach((id, i) => { const node = coordinates[id]; if (node) i ? context.lineTo(sx(node.x), sy(node.y)) : context.moveTo(sx(node.x), sy(node.y)); });
    context.stroke();
  }
  osmPreviewNodes.forEach((node) => { node.px = sx(node.x); node.py = sy(node.y); context.fillStyle = "#fff"; context.beginPath(); context.arc(node.px, node.py, 3, 0, Math.PI * 2); context.fill(); });
}

async function loadOSM(names) {
  const status = document.querySelector("#osm-status");
  const origins = document.querySelector("#osm-origins").value.split(",").map((v) => v.trim()).filter(Boolean);
  const shelters = document.querySelector("#osm-shelters").value.split(",").map((v) => v.trim()).filter(Boolean);
  const agents = Number(document.querySelector("#osm-agents").value);
  if (!osmDocument || !origins.length || !shelters.length || agents < 1) {
    status.textContent = "Choose a file, origins, shelters, and population.";
    return;
  }
  panels.classList.add("loading");
  summary.hidden = true;
  setPlaying(false);
  status.textContent = "Importing roads and running simulations…";
  try {
    const query = new URLSearchParams({
      routers: names.join(","), origins: origins.join(","),
      shelters: shelters.join(","), agents: String(agents),
      capacity_multiplier: document.querySelector("#osm-capacity").value,
      name: document.querySelector("#osm-file").files[0]?.name || "OSM map",
      disaster: document.querySelector("#hazard").value,
    });
    const response = await fetch(`/api/osm/simulation?${query}`, {
      method: "POST", headers: {"Content-Type": "application/xml"}, body: osmDocument,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "OSM import failed");
    data = payload;
    index = 0;
    updateScenarioLabel();
    rebuild();
    status.textContent = `Loaded ${data.configuration.import.imported_nodes} road nodes.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    panels.classList.remove("loading");
  }
}

function updateScenarioLabel() {
  const config = data.configuration;
  document.querySelector("#scenario").textContent = data.map_type === "openstreetmap"
    ? `${config.source_name}  /  ${config.import.imported_nodes.toLocaleString()} ROAD NODES  /  ${config.agent_count.toLocaleString()} PEOPLE`
    : `${config.width}×${config.height} GRID  /  ${config.agent_count.toLocaleString()} PEOPLE  /  ${config.disaster_profile.toUpperCase()}`;
}

function renderMap(canvas, run, frame) {
  const dpr = devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  context.clearRect(0, 0, width, height);
  const nodes = Object.fromEntries(run.topology.nodes.map((node) => [node.id, node]));
  const xs = run.topology.nodes.map((node) => node.x);
  const ys = run.topology.nodes.map((node) => node.y);
  const pad = 34;
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (x) => pad + (x - minX) / (maxX - minX || 1) * (width - pad * 2);
  const sy = (y) => pad + (y - minY) / (maxY - minY || 1) * (height - pad * 2);
  const edgeState = Object.fromEntries(
    frame.edges.map((edge) => [`${edge.source}>${edge.target}`, edge]),
  );
  context.lineCap = "round";

  for (const edge of run.topology.edges) {
    const start = nodes[edge.source];
    const end = nodes[edge.target];
    const forward = edgeState[`${edge.source}>${edge.target}`];
    const reverse = edgeState[`${edge.target}>${edge.source}`];
    const state = forward || reverse;
    const load = Math.max(forward?.occupancy || 0, reverse?.occupancy || 0) / edge.capacity;
    context.strokeStyle = state?.blocked ? colors.blocked
      : state?.hazard ? colors.hazard
      : load ? `rgba(8,126,164,${0.35 + 0.65 * load})` : colors.road;
    context.lineWidth = state?.blocked ? 3 : 1 + load * 5;
    context.beginPath();
    context.moveTo(sx(start.x), sy(start.y));
    context.lineTo(sx(end.x), sy(end.y));
    context.stroke();
  }

  for (const node of frame.nodes) {
    const position = nodes[node.id];
    context.fillStyle = `rgba(226,35,26,${Math.min(0.78, 0.18 + node.hazard * 0.35)})`;
    context.beginPath();
    context.arc(sx(position.x), sy(position.y), 11 + node.hazard * 4, 0, Math.PI * 2);
    context.fill();
  }

  context.font = "600 11px -apple-system, BlinkMacSystemFont, sans-serif";
  context.textAlign = "center";
  for (const shelter of frame.shelters) {
    const position = nodes[shelter.node];
    const x = sx(position.x);
    const y = sy(position.y);
    context.fillStyle = shelter.full ? colors.blocked : colors.shelter;
    context.fillRect(x - 7, y - 7, 14, 14);
    context.fillStyle = "#ffffff";
    context.fillRect(x - 2, y - 6, 4, 12);
    context.fillRect(x - 6, y - 2, 12, 4);
    const label = shelter.capacity === 0
      ? "CLOSED"
      : `${shelter.occupants}/${shelter.capacity}`;
    context.fillStyle = shelter.full ? colors.blocked : "#ffffff";
    context.fillText(label, x, y - 12);
  }

  for (const agent of frame.agents) {
    if (agent.status === "evacuated") continue;
    context.fillStyle = agent.status === "stranded" ? colors.blocked
      : agent.status === "waiting" ? colors.waiting : colors.agent;
    context.globalAlpha = 0.75;
    context.beginPath();
    context.arc(sx(agent.x), sy(agent.y), 2.2, 0, Math.PI * 2);
    context.fill();
  }
  context.globalAlpha = 1;
}

function routingText(reasons) {
  const entries = Object.entries(reasons)
    .filter(([reason]) => reason !== "initial route")
    .sort((a, b) => b[1] - a[1]);
  if (!entries.length) return "No affected agents yet";
  return `Affected agents: ${entries.slice(0, 3).map(([reason, count]) => `${reason} ${count}`).join(" · ")}`;
}

function formatDuration(ticks) {
  if (data.map_type !== "openstreetmap") return `${ticks} ticks`;
  const seconds = Math.round(ticks * (data.configuration.tick_seconds || 1));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  const parts = [];
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  if (remainder || !parts.length) parts.push(`${remainder}s`);
  return `${ticks.toLocaleString()} ticks (${parts.join(" ")})`;
}

function winner(metric, formatter, lowerIsBetter = true) {
  const [first, second] = data.runs;
  if (!second) return `${first.algorithm}: ${formatter(first.result[metric])}`;
  const a = first.result[metric];
  const b = second.result[metric];
  if (Math.abs(a - b) < 1e-12) return `Tie: ${formatter(a)}`;
  const winning = (lowerIsBetter ? a < b : a > b) ? first : second;
  return `${winning.algorithm}: ${formatter(winning.result[metric])}`;
}

function renderSummary() {
  summary.hidden = index < maxIndex();
  if (summary.hidden) return;
  const metrics = [
    ["Fastest evacuation", winner("elapsed_ticks", formatDuration)],
    ["Lowest exposure", winner("mean_hazard_exposure", (value) => value.toFixed(2))],
    ["Least waiting", winner("mean_waiting_time", (value) => value.toFixed(2))],
    ["Most equal completion", winner("completion_time_gini", (value) => `Gini ${value.toFixed(3)}`)],
  ];
  document.querySelector("#summary-content").innerHTML = `<div class="summary-grid">${metrics.map(
    ([label, value]) => `<div><strong>${label}</strong><span>${value}</span></div>`,
  ).join("")}</div>`;
}

function draw() {
  document.querySelector("#tick").value = data.map_type === "openstreetmap"
    ? `TICK ${String(currentTick()).padStart(3, "0")} · ${formatDuration(currentTick()).replace(/^.*? \(/, "").replace(/\)$/, "")}`
    : `TICK ${String(currentTick()).padStart(3, "0")}`;
  document.querySelector("#timeline").value = index;
  data.runs.forEach((run, panelIndex) => {
    const panel = panels.children[panelIndex];
    const frame = run.frames[Math.min(index, run.frames.length - 1)];
    const count = frame.counts;
    const total = data.configuration.agent_count;
    const finished = count.evacuated + count.stranded;
    renderMap(panel.querySelector("canvas"), run, frame);
    panel.querySelector('[data-key="evacuated"]').textContent = `${count.evacuated.toLocaleString()} / ${total.toLocaleString()}`;
    panel.querySelector('[data-key="active"]').textContent = (count.waiting + count.traveling).toLocaleString();
    panel.querySelector('[data-key="exposure"]').textContent = frame.mean_hazard_exposure.toFixed(2);
    panel.querySelector('[data-key="final"]').textContent = formatDuration(run.result.elapsed_ticks);
    panel.querySelector('[data-key="routing"]').textContent = routingText(frame.routing_reasons);
    panel.querySelector(".progress span").style.width = `${finished / total * 100}%`;
    panel.querySelector(".status").textContent = finished === total ? "COMPLETE" : "RUNNING";
  });
  renderSummary();
}

function currentTick() {
  return Math.max(...data.runs.map((run) => run.frames[Math.min(index, run.frames.length - 1)].tick));
}
function setPlaying(value) {
  playing = value;
  document.querySelector("#play").textContent = playing ? "Pause" : "Play";
}
function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  timer = setTimeout(() => {
    index = Math.min(index + 1, maxIndex());
    draw();
    if (index >= maxIndex()) {
      setPlaying(false);
      document.querySelector("#replay").hidden = false;
    } else schedule();
  }, Number(document.querySelector("#speed").value));
}
function maxIndex() {
  return Math.max(...data.runs.map((run) => run.frames.length)) - 1;
}

Promise.all([
  fetch("/api/config").then((response) => response.json()),
  fetch("/api/simulation").then((response) => response.json()),
]).then(([config, payload]) => {
  routerOptions = config.routers;
  disasterOptions = config.disasters;
  data = payload;
  updateScenarioLabel();
  const hazard = document.querySelector("#hazard");
  for (const name of disasterOptions) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name === "none" ? "None" : name.replace("-", " ");
    option.selected = name === data.configuration.disaster_profile;
    hazard.append(option);
  }
  hazard.onchange = () => loadScenario(data.runs.map((run) => run.algorithm), hazard.value);
  const osmFile = document.querySelector("#osm-file");
  osmFile.onchange = async () => {
    const file = osmFile.files[0];
    if (!file) return;
    if (file.size > 20000000) { document.querySelector("#osm-status").textContent = "The local file limit is 20 MB."; return; }
    osmDocument = await file.text();
    try { previewOSM(osmDocument); document.querySelector("#osm-status").textContent = "Click a node to assign it."; }
    catch (error) { document.querySelector("#osm-status").textContent = error.message; }
  };
  document.querySelector("#osm-preview").onclick = (event) => {
    if (!osmPreviewNodes.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
    const nearest = osmPreviewNodes.reduce((best, node) =>
      Math.hypot(node.px - x, node.py - y) < Math.hypot(best.px - x, best.py - y) ? node : best
    );
    const mode = document.querySelector("#osm-pick-mode").value;
    const field = document.querySelector(mode === "origins" ? "#osm-origins" : "#osm-shelters");
    const values = field.value.split(",").map((value) => value.trim()).filter(Boolean);
    if (!values.includes(nearest.id)) values.push(nearest.id);
    field.value = values.join(",");
    document.querySelector("#osm-status").textContent = `Selected node ${nearest.id} as ${mode === "origins" ? "an origin" : "a shelter"}.`;
  };
  document.querySelector("#load-osm").onclick = async () => {
    const file = document.querySelector("#osm-file").files[0];
    if (!file) {
      document.querySelector("#osm-status").textContent = "Choose a local .osm file first.";
      return;
    }
    if (file.size > 20000000) {
      document.querySelector("#osm-status").textContent = "The local file limit is 20 MB.";
      return;
    }
    osmDocument = await file.text();
    await loadOSM(data.runs.map((run) => run.algorithm));
  };
  const slider = document.querySelector("#timeline");
  slider.addEventListener("input", (event) => {
    index = Number(event.target.value);
    document.querySelector("#replay").hidden = index < maxIndex();
    draw();
  });
  document.querySelector("#play").onclick = () => {
    if (!playing && index >= maxIndex()) {
      index = 0;
      document.querySelector("#replay").hidden = true;
    }
    setPlaying(!playing);
    draw();
    schedule();
  };
  document.querySelector("#replay").onclick = () => {
    index = 0;
    document.querySelector("#replay").hidden = true;
    setPlaying(true);
    draw();
    schedule();
  };
  document.querySelector("#speed").onchange = schedule;
  addEventListener("resize", draw);
  rebuild();
}).catch(() => {
  panels.textContent = "The simulation data could not be loaded.";
});
