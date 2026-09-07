const panels = document.querySelector("#panels");
const template = document.querySelector("#panel-template");
const summary = document.querySelector("#summary");
let data;
let index = 0;
let playing = false;
let timer;
let routerOptions = [];
let disasterOptions = [];

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

function updateScenarioLabel() {
  const config = data.configuration;
  document.querySelector("#scenario").textContent =
    `${config.width}×${config.height} GRID  /  ${config.agent_count.toLocaleString()} PEOPLE  /  ${config.disaster_profile.toUpperCase()}`;
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
  const sx = (x) => pad + x / (Math.max(...xs) || 1) * (width - pad * 2);
  const sy = (y) => pad + y / (Math.max(...ys) || 1) * (height - pad * 2);
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
  if (!entries.length) return "No rerouting decisions yet";
  return `Rerouting: ${entries.slice(0, 3).map(([reason, count]) => `${reason} ${count}`).join(" · ")}`;
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
    ["Fastest evacuation", winner("elapsed_ticks", (value) => `${value} ticks`)],
    ["Lowest exposure", winner("mean_hazard_exposure", (value) => value.toFixed(2))],
    ["Least waiting", winner("mean_waiting_time", (value) => value.toFixed(2))],
    ["Most equal completion", winner("completion_time_gini", (value) => `Gini ${value.toFixed(3)}`)],
  ];
  document.querySelector("#summary-content").innerHTML = `<div class="summary-grid">${metrics.map(
    ([label, value]) => `<div><strong>${label}</strong><span>${value}</span></div>`,
  ).join("")}</div>`;
}

function draw() {
  document.querySelector("#tick").value = `TICK ${String(currentTick()).padStart(3, "0")}`;
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
    panel.querySelector('[data-key="final"]').textContent = `${run.result.elapsed_ticks} ticks`;
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
