"use strict";
const byId = id => document.getElementById(id);
const cells = [];
let currentState = null;
let busy = false;
let configDirty = false;
let searchMode = false;
function updateConfigInputs() {
  const rotation = byId("config-round-robin").checked;
  const sides = rotation ? ["one", "two", "three"] : ["one", "two"];
  const models = sides.map(side => byId(`config-player-${side}`).value === "rules" ? "rules" : byId(`config-model-${side}`).value.trim());
  const invalid = rotation ? models.filter(model => model === "rules").length > 1 : models.every(model => model === "rules");
  byId("config-third-player").hidden = !rotation;
  byId("config-round-robin").disabled = busy;
  byId("config-curriculum").disabled = busy;
  for (const side of ["one", "two", "three"]) {
    const rules = byId(`config-player-${side}`).value === "rules";
    byId(`config-model-${side}-label`).hidden = rules;
    byId(`config-model-${side}`).disabled = busy || rules || (side === "three" && !rotation);
    byId(`config-model-${side}`).setCustomValidity(!rules && byId(`config-model-${side}`).value.trim() === "rules"
      ? "The name rules is reserved for the rule-based opponent. Choose a different AI model name." : "");
    byId(`config-player-${side}`).disabled = busy || (side === "three" && !rotation);
  }
  byId("config-swap-sides").disabled = busy;
  byId("apply-simulation-config").disabled = busy || invalid;
  byId("apply-simulation-config").textContent = invalid ? (rotation ? "Choose at most one rule player" : "Choose at least one AI") : "Apply configuration";
}
function renderConfig(config) {
  if (!config) return;
  if (!configDirty) {
    byId("config-round-robin").checked = Boolean(config.round_robin);
    byId("config-curriculum").checked = config.curriculum !== false;
    for (const side of ["one", "two", "three"]) {
      const model = config[`model_${side}`];
      if (!model) continue;
      byId(`config-player-${side}`).value = model === "rules" ? "rules" : "ai";
      if (model !== "rules") byId(`config-model-${side}`).value = model;
    }
    byId("config-swap-sides").checked = config.swap_sides;
  }
  const roster = [config.model_one, config.model_two, ...(config.round_robin ? [config.model_three] : [])];
  byId("config-summary").textContent = `Applied: ${roster.map(actorName).join(" / ")}. ${config.round_robin ? "Round robin, 15 games per pair. " : ""}Switching is ${config.swap_sides ? "on" : "off"}.`;
  updateConfigInputs();
}
const columnButtons = Array.from({length: 7}, (_, column) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = column;
  button.setAttribute("aria-label", `Drop a disc in column ${column}`);
  button.disabled = true;
  button.addEventListener("click", () => playColumn(column));
  document.querySelector(".columns").appendChild(button);
  return button;
});
for (let row = 0; row < 5; row++) {
  for (let column = 0; column < 7; column++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.addEventListener("click", () => playColumn(column));
    byId("board").appendChild(cell);
    cells.push({cell, row, column});
  }
}
const number = value => Number(value || 0).toLocaleString();
const actorName = model => model === "rules" ? "Rule-based opponent" : model === "human" ? "Human" : `AI · ${model}`;
let lastFrame = "";
function connection(label, live) {
  const badge = byId("connection");
  badge.className = `badge ${live ? "live" : "offline"}`;
  badge.replaceChildren(Object.assign(document.createElement("i")), document.createTextNode(label));
}
function render(state) {
  if (currentState && state.updated_at < currentState.updated_at) return;
  currentState = state;
  renderConfig(state.config);
  const human = state.mode === "human";
  byId("rotation-status").hidden = !state.rotation;
  if (state.rotation) byId("rotation-status").textContent = `${human ? "Paused" : "Round " + state.rotation.round}: slots ${state.rotation.slots.join(" vs ")} (${state.rotation.pair.map(actorName).join(" vs ")}) · Game ${state.rotation.game} / 15${state.status === "finished" ? " completed" : ""}`;
  const fresh = Date.now() / 1000 - state.updated_at < 10;
  const live = human ? state.status !== "human_error" : fresh && ["playing", "finished"].includes(state.status);
  connection(human ? "Human play · simulation paused" : live ? "Live training" : state.status === "reconnecting" ? "Reconnecting" : "Waiting for moves", live);
  byId("simulation-mode").setAttribute("aria-pressed", String(!human));
  byId("human-mode").setAttribute("aria-pressed", String(human));
  byId("new-game").hidden = !human;
  byId("retry-bot").hidden = !human || state.status !== "human_error";
  document.querySelectorAll(".mode-controls button").forEach(button => { button.disabled = busy; });
  byId("mode-note").textContent = human
    ? `You are yellow and go first. Click a column to play against ${state.model_two}. The bot keeps learning; simulation is paused.`
    : `${[state.model_one, state.model_two].includes("rules") ? "The AI trains against rules." : "Both AI players are training."} ${state.config?.swap_sides ? "Players swap sides after each game." : "Player positions stay fixed."} Switch to human play to pause simulation.`;
  if (searchMode) {
    connection(human ? "Human play · background learning continues" : "Published AI exhibition", live);
    byId("mode-note").textContent = human
      ? "You play the published model. Its confirmed game outcome enters replay; background practice continues independently."
      : "This is an inference-only exhibition using published weights. Background games train a separate candidate. Open the debug console for live training and publication status.";
  }
  columnButtons.forEach((button, column) => {
    button.disabled = busy || !human || state.done || state.next_player !== 1
      || state.status !== "human_turn" || state.board[column][0] !== 0;
  });
  byId("board").classList.toggle("playable", human && !busy && state.status === "human_turn");
  const frame = `${state.game_id}:${state.move_count}`;
  if (frame !== lastFrame) {
    const lastRow = state.last_column == null ? -1 : state.board[state.last_column].findIndex(v => v !== 0);
    for (const {cell, row, column} of cells) {
      const value = state.board[column][row];
      cell.className = "cell" + (value === 1 ? " yellow" : value === -1 ? " red" : "")
        + (column === state.last_column && row === lastRow ? " latest" : "");
      cell.title = `Column ${column}, row ${row}: ${value === 0 ? "empty" : `player ${value}`}`;
    }
    byId("board").setAttribute("aria-label", `Game board after ${state.move_count} moves. `
      + state.board.map((col, i) => `Column ${i}: ${col.join(", ")}`).join(". "));
    columnButtons.forEach((el, i) => el.classList.toggle("last", i === state.last_column));
    lastFrame = frame;
  }
  byId("game-number").textContent = state.game_id ? `${human ? "Human game" : "Game"} ${number(state.games + (state.status === "finished" ? 0 : 1))}` : "Waiting for a game";
  byId("move-number").textContent = `Move ${state.move_count} / 35`;
  byId("model-one").textContent = human ? "You · human" : actorName(state.model_one);
  byId("model-two").textContent = actorName(state.model_two);
  byId("model-one").title = human ? "You" : actorName(state.model_one);
  byId("model-two").title = actorName(state.model_two);
  byId("player-one").classList.toggle("active", !state.done && state.next_player === 1);
  byId("player-two").classList.toggle("active", !state.done && state.next_player === -1);
  byId("game-status").textContent = human
    ? state.status === "human_error" ? "Bot action failed. Retry or start a new game."
      : state.done ? (state.winner === 0 ? "A draw! Play again with New game."
        : state.winner === 1 ? "You win! Play again with New game." : "The bot wins! Play again with New game.")
      : busy || state.next_player === -1 ? "Bot is thinking and training…" : "Your turn — choose a column"
    : state.status === "reconnecting" ? "Waiting for the model API…"
    : state.status === "stopped" ? "Simulation stopped"
    : state.done ? (state.winner === 0 ? "A draw. Next game shortly." : `${actorName(state.winner === 1 ? state.model_one : state.model_two)} wins!`)
    : state.game_id ? `${actorName(state.next_player === 1 ? state.model_one : state.model_two)} to move` : "Waiting for the simulator…";
  byId("last-move").textContent = state.last_column == null ? "—" : `Last disk → column ${state.last_column}`;
  byId("move-reason").textContent = state.move_reason ? `${actorName(state.actor)}: ${state.move_reason}` : "";
  byId("device").textContent = state.device || "—";
  byId("updates").textContent = number(state.updates);
  byId("replay").replaceChildren(document.createTextNode(number(state.replay_size) + " "),
    Object.assign(document.createElement("small"), {textContent: "/ 20,000"}));
  byId("epsilon").textContent = state.epsilon == null ? "—" : `${(state.epsilon * 100).toFixed(1)}%`;
  byId("loss").textContent = state.loss == null ? "Warming up" : state.loss.toFixed(5);
  const diagnostic = state.diagnostics || {};
  byId("td-loss").textContent = diagnostic.td_loss == null ? "Warming up" : diagnostic.td_loss.toFixed(5);
  byId("tactical-loss").textContent = diagnostic.tactical_loss == null ? "Warming up" : diagnostic.tactical_loss.toFixed(5);
  byId("q-range").textContent = diagnostic.q_min == null ? "Waiting" : `${diagnostic.q_min.toFixed(2)} to ${diagnostic.q_max.toFixed(2)}`;
  byId("rule-strength").textContent = `${Math.round((state.rule_probability ?? 1) * 100)}%`;
  byId("api-errors").textContent = number(state.api_errors);
  const warnings = [];
  if (diagnostic.q_warning) warnings.push("Predicted rewards exceed the expected -1 to +1 range.");
  if (diagnostic.checkpoint_error) warnings.push("Checkpoint save failed; learning continues in memory and saving will retry.");
  if (state.service_errors?.training) warnings.push(`${state.service_errors.training} training errors recorded.`);
  byId("training-warning").hidden = warnings.length === 0;
  byId("training-warning").textContent = warnings.join(" ");
  byId("interval").textContent = `${state.interval}s`;
  byId("last-model").textContent = state.model_id || "—";
  if (searchMode) {
    byId("replay").textContent = `${number(state.replay_size)} / 100,000`;
    byId("loss").textContent = diagnostic.policy_value_loss == null ? "See debug console" : diagnostic.policy_value_loss.toFixed(4);
    byId("epsilon").textContent = "Search exploration (training only)";
    byId("td-loss").textContent = "Not used by this engine";
    byId("tactical-loss").textContent = "Not used by this engine";
    byId("q-range").textContent = "See move inspector";
  }
  byId("games").textContent = `${number(state.games)} games`;
  byId("wins-one").textContent = number(state.scores["1"]);
  byId("wins-two").textContent = number(state.scores["-1"]);
  byId("draws").textContent = number(state.scores["0"]);
  renderMatchups(state);
  byId("game-id").textContent = state.game_id || "No active game";
  byId("error").hidden = !state.error;
  byId("error").textContent = state.error || "";
}
function renderMatchups(state) {
  const config = state.config;
  const models = [...new Set([config.model_one, config.model_two, ...(config.round_robin ? [config.model_three] : [])])];
  const table = document.createElement("table");
  const caption = table.createCaption();
  caption.textContent = "Row starts · column plays second";
  const header = table.createTHead().insertRow();
  const corner = document.createElement("th");
  corner.textContent = "First / second";
  header.append(corner);
  for (const model of models) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = actorName(model);
    header.append(cell);
  }
  const body = table.createTBody();
  for (const first of models) {
    const row = body.insertRow();
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = actorName(first);
    row.append(label);
    for (const second of models) {
      const counts = state.matchups?.[first]?.[second] || {"1": 0, "0": 0, "-1": 0};
      const cell = row.insertCell();
      cell.textContent = `${number(counts["1"])} / ${number(counts["0"])} / ${number(counts["-1"])}`;
      cell.title = `${actorName(first)} starts against ${actorName(second)}: ${counts["1"]} first-player wins, ${counts["0"]} draws, ${counts["-1"]} second-player wins`;
    }
  }
  byId("matchup-matrix").replaceChildren(table);
}
async function command(path, payload = {}) {
  if (busy) return;
  busy = true;
  byId("control-error").hidden = true;
  if (currentState) render(currentState);
  try {
    const response = await fetch(path, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail
      : Array.isArray(result.detail) ? result.detail.map(error => error.msg).join("; ") : `HTTP ${response.status}`);
    if (path === "/simulation/config") configDirty = false;
    render(result);
  } catch (error) {
    byId("control-error").textContent = error.message;
    byId("control-error").hidden = false;
  } finally {
    busy = false;
    if (currentState) render(currentState);
  }
}
function playColumn(column) {
  if (!currentState || columnButtons[column].disabled) return;
  command("/human/move", {game_id: currentState.game_id, move_count: currentState.move_count, column});
}
byId("simulation-mode").addEventListener("click", () => command("/mode", {mode: "simulation"}));
byId("human-mode").addEventListener("click", () => command("/mode", {mode: "human"}));
byId("new-game").addEventListener("click", () => command("/human/new"));
byId("retry-bot").addEventListener("click", () => command("/human/retry", {
  game_id: currentState.game_id, move_count: currentState.move_count,
}));
byId("simulation-config").addEventListener("input", () => {
  configDirty = true;
  updateConfigInputs();
});
byId("simulation-config").addEventListener("submit", event => {
  event.preventDefault();
  command("/simulation/config", {
    model_one: byId("config-player-one").value === "rules" ? "rules" : byId("config-model-one").value.trim(),
    model_two: byId("config-player-two").value === "rules" ? "rules" : byId("config-model-two").value.trim(),
    swap_sides: byId("config-swap-sides").checked,
    round_robin: byId("config-round-robin").checked,
    curriculum: byId("config-curriculum").checked,
    model_three: byId("config-round-robin").checked
      ? (byId("config-player-three").value === "rules" ? "rules" : byId("config-model-three").value.trim()) : null,
  });
});
let trainingBusy = false;
function svgElement(tag, attributes = {}, text = null) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}
function strengthChart(history, model) {
  const chart = svgElement("svg", {viewBox: "0 0 560 185", role: "img",
    "aria-label": `${model}: benchmark score by optimizer updates`});
  const first = history[0], last = history[history.length - 1];
  const x = point => 42 + 500 * (point.updates - first.updates) / Math.max(1, last.updates - first.updates);
  const y = point => 145 - 130 * point.score;
  for (const percent of [0, 50, 100]) {
    const height = 145 - 1.3 * percent;
    chart.append(svgElement("line", {x1: 42, y1: height, x2: 542, y2: height, class: "chart-grid"}));
    chart.append(svgElement("text", {x: 33, y: height + 4, "text-anchor": "end"}, `${percent}%`));
  }
  chart.append(svgElement("polyline", {points: history.map(point => `${x(point)},${y(point)}`).join(" "), class: "chart-line"}));
  for (const point of history) {
    const dot = svgElement("circle", {cx: x(point), cy: y(point), r: 3.5, class: "chart-dot"});
    dot.append(svgElement("title", {}, `Update ${number(point.updates)}: score ${(point.score * 100).toFixed(1)}%; ${point.wins} wins, ${point.draws} draws, ${point.losses} losses`));
    chart.append(dot);
  }
  chart.append(svgElement("text", {x: 42, y: 165}, number(first.updates)));
  if (last.updates !== first.updates) chart.append(svgElement("text", {x: 542, y: 165, "text-anchor": "end"}, number(last.updates)));
  chart.append(svgElement("text", {x: 292, y: 181, "text-anchor": "middle"}, "Optimizer updates"));
  return chart;
}
function renderProgress(progress) {
  const hasSearch = Object.values(progress.models).some(model => model.architecture?.startsWith("policy-value"));
  searchMode = hasSearch;
  if (hasSearch) {
    byId("simulation-settings-title").textContent = "Exhibition configuration";
    const notes = byId("simulation-config").querySelectorAll(":scope > p.note");
    if (notes[0]) notes[0].textContent = "This option changes only the displayed rule opponent. Background practice always uses its own 60% recent AI / 25% rules / 15% random mixture.";
    if (notes[notes.length-1]) notes[notes.length-1].textContent = "These settings change exhibition games, not the background learning schedule. Use shared for visitor games so confirmed outcomes contribute to the same learner.";
    if (currentState) render(currentState);
  }
  document.querySelector(".exploration-presets").hidden = hasSearch;
  const cards = [];
  for (const model of Object.values(progress.models)) {
    const history = model.history;
    const latest = history[history.length - 1] || {score: 0, wins: 0, draws: 0, losses: 0, win_rate: 0};
    const searchBackend = model.architecture?.startsWith("policy-value");
    const card = document.createElement("article");
    card.className = "progress-card";
    const title = document.createElement("h3");
    title.textContent = model.model_id;
    const architecture = document.createElement("p");
    architecture.className = "note";
    architecture.textContent = `Tiny CNN · 7 × 5 board input · ${number(model.observations)} observations`;
    if (model.pretraining) architecture.textContent += ` · Warm-started from ${number(model.pretraining.positions)} rule-labeled positions`;
    if (model.pretraining?.win_accuracy != null) architecture.textContent +=
      ` · Warm-start test: ${(model.pretraining.win_accuracy * 100).toFixed(1)}% immediate wins, ${(model.pretraining.block_accuracy * 100).toFixed(1)}% blocks`;
    if (searchBackend) architecture.textContent = `Learning move preferences and position strength · ${number(model.observations)} observations · Published generation ${number(model.diagnostics?.generation)}`;
    const summary = document.createElement("p");
    summary.className = "progress-score";
    const score = document.createElement("strong");
    score.textContent = history.length ? `${(latest.score * 100).toFixed(1)}%` : "Measuring…";
    const label = document.createElement("span");
    const change = history.length ? (latest.score - history[0].score) * 100 : 0;
    label.textContent = history.length > 1 ? `benchmark score · ${change >= 0 ? "+" : ""}${change.toFixed(1)} points since first shown check` : "benchmark score · starting measurement";
    summary.append(score, label);
    const detail = document.createElement("p");
    detail.className = "progress-detail";
    detail.textContent = `${latest.wins}W / ${latest.draws}D / ${latest.losses}L · Win rate ${(latest.win_rate * 100).toFixed(1)}% · Discs per win ${latest.winning_discs == null ? "—" : latest.winning_discs.toFixed(1)}`;
    if (latest.positions) detail.textContent += ` · Starts first: ${latest.positions.first.wins}W/${latest.positions.first.draws}D/${latest.positions.first.losses}L · Starts second: ${latest.positions.second.wins}W/${latest.positions.second.draws}D/${latest.positions.second.losses}L`;
    if (latest.tactics) detail.textContent += ` · Immediate wins: ${(latest.tactics.win_accuracy * 100).toFixed(1)}% · Blocks: ${(latest.tactics.block_accuracy * 100).toFixed(1)}%`;
    if (latest.mean_return != null) detail.textContent += ` · Mean discounted game reward: ${latest.mean_return.toFixed(3)}`;
    if (searchBackend && history.length) detail.textContent += ` · ${latest.stage === "baseline" ? "Starting model" : latest.promoted ? "Candidate published" : "Candidate kept training; published model unchanged"}`;
    if (latest.arena) detail.textContent += ` · Against published model: ${(latest.arena.score * 100).toFixed(1)}% over ${latest.arena.games} games`;
    if (!history.length) detail.textContent = "The first strength check runs in the background.";
    const diagnostics = document.createElement("p");
    diagnostics.className = "note";
    const d = model.diagnostics || {};
    diagnostics.textContent = `Training errors: ${number(model.service_errors?.training)} · Checkpoint errors: ${number(model.service_errors?.checkpoint)} · Evaluation errors: ${number(model.service_errors?.evaluation)} · Request errors: ${number(model.service_errors?.requests)}`;
    if (d.q_min != null) diagnostics.textContent += ` · Sampled Q: ${d.q_min.toFixed(2)} to ${d.q_max.toFixed(2)} · Targets: ${d.target_min.toFixed(2)} to ${d.target_max.toFixed(2)}`;
    if (model.last_service_error) diagnostics.textContent += ` · Last error: ${model.last_service_error}`;
    if (model.background?.error) diagnostics.textContent += ` · Background learner stopped: ${model.background.error}`;
    const exploration = document.createElement("p");
    exploration.className = "note";
    exploration.textContent = `Simulation exploration: ${(model.exploration_rate * 100).toFixed(1)}% · ${number(model.exploration_decisions)} exploratory decisions since reset. Human games use 0%.`;
    if (searchBackend) exploration.textContent = `Background learner ${model.background?.running ? "running" : "stopped"} · ${number(model.background?.games)} completed training games this session · ${number(model.background?.replay_size)} replay positions · ${number(model.background?.human_replay_size)} human positions. Search explores training openings; physical games choose the most visited move.`;
    const freshness = document.createElement("p");
    freshness.className = "note";
    freshness.textContent = `Measured at update ${number(latest.updates)} · Current update ${number(model.updates)} · ${new Date(latest.timestamp * 1000).toLocaleString()}`;
    if (!history.length) freshness.textContent = `Current update ${number(model.updates)} · Waiting for the first benchmark`;
    card.append(title, architecture, summary);
    if (history.length) card.append(strengthChart(history, model.model_id));
    card.append(detail, diagnostics, exploration, freshness);
    const frame = model.background?.latest_game;
    if (frame) {
      const caption = document.createElement("p");
      caption.className = "note";
      caption.textContent = `Live background training sample: AI vs ${frame.opponent} · ${frame.move_count} discs`;
      const board = document.createElement("div");
      board.className = "training-sample";
      board.setAttribute("aria-label", "Live background training board");
      for (let row = 0; row < 5; row++) for (let col = 0; col < 7; col++) {
        const cell = document.createElement("span");
        cell.className = frame.board[col][row] === 1 ? "yellow" : frame.board[col][row] === -1 ? "red" : "empty";
        board.append(cell);
      }
      card.append(caption, board);
    }
    cards.push(card);
  }
  byId("progress-models").replaceChildren(...cards);
}
async function trainingCommand(path, label, payload) {
  if (trainingBusy) return;
  trainingBusy = true;
  document.querySelectorAll("[data-exploration-rate], #measure-progress").forEach(button => { button.disabled = true; });
  byId("progress-message").textContent = label;
  try {
    const options = {method: "POST"};
    if (payload !== undefined) {
      options.headers = {"Content-Type": "application/json"};
      options.body = JSON.stringify(payload);
    }
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    renderProgress(result);
    byId("progress-message").textContent = path === "/exploration/reset"
      ? `Exploration restarted at ${Math.round(payload.rate * 100)}% for the displayed models. Weights and experience are preserved.`
      : "Measurement requested. Results appear when the background check finishes.";
  } catch (error) {
    byId("progress-message").textContent = error.message;
  } finally {
    trainingBusy = false;
    document.querySelectorAll("[data-exploration-rate], #measure-progress").forEach(button => { button.disabled = false; });
  }
}
document.querySelectorAll("[data-exploration-rate]").forEach(button => {
  button.addEventListener("click", () => {
    const rate = Number(button.dataset.explorationRate);
    trainingCommand("/exploration/reset", `Resetting exploration to ${Math.round(rate * 100)}%…`, {rate});
  });
});
byId("measure-progress").addEventListener("click", () => trainingCommand("/evaluation", "Requesting a strength check…"));
async function pollProgress() {
  try {
    if (!trainingBusy) {
      const response = await fetch("/progress", {cache: "no-store", signal: AbortSignal.timeout(60000)});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const progress = await response.json();
      if (!trainingBusy) {
        renderProgress(progress);
        byId("progress-message").textContent = "Progress refreshes every 5 seconds. Measurements can rise or fall as the bot learns.";
      }
    }
  } catch (error) {
    byId("progress-message").textContent = "Progress unavailable. Retrying automatically.";
  } finally {
    setTimeout(pollProgress, 5000);
  }
}
async function poll() {
  try {
    const response = await fetch("/state", {cache: "no-store", signal: AbortSignal.timeout(4000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    connection("Viewer disconnected", false);
    byId("game-status").textContent = "Connection lost. Retrying…";
  } finally {
    setTimeout(poll, 250);
  }
}
poll();
pollProgress();
