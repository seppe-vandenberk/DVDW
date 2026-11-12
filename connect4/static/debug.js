"use strict";
const el = id => document.getElementById(id);
const fmt = n => n == null ? "—" : Number(n).toLocaleString(undefined, {maximumFractionDigits: 2});
const pct = n => n == null ? "—" : `${(n * 100).toFixed(1)}%`;
const time = t => t ? new Date(t * 1000).toLocaleTimeString() : "Not yet";
const age = t => t ? `${Math.max(0, Math.round(Date.now() / 1000 - t))}s ago` : "Not yet";
const empty = () => Array.from({length: 7}, () => Array(5).fill(0));
let state = null, selectedModel = "shared", commanding = false;
function node(tag, text, className) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (className) n.className = className;
  return n;
}
function facts(id, entries) {
  el(id).replaceChildren(...entries.map(([label, value, style]) => {
    const row = node("div"); row.append(node("dt", label), node("dd", value, style)); return row;
  }));
}
function rows(id, data, blank, columns) {
  const result = data.map(values => {
    const row = node("tr");
    for (const value of values) {
      const [text, style] = Array.isArray(value) ? value : [value];
      row.append(node("td", text, style));
    }
    return row;
  });
  if (!result.length) {
    const row = node("tr"), cell = node("td", blank, "muted");
    cell.colSpan = columns; row.append(cell); result.push(row);
  }
  el(id).replaceChildren(...result);
}
function board(id, data) {
  const cells = [];
  for (let r = 0; r < 5; r++) for (let c = 0; c < 7; c++) {
    const value = data?.[c]?.[r] || 0;
    const cell = node("span", null, value === 1 ? "yellow" : value === -1 ? "red" : "empty");
    cell.title = `Column ${c}, row ${r}: ${value}`; cells.push(cell);
  }
  el(id).replaceChildren(...cells);
}
function svgNode(tag, attrs, text) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value);
  if (text != null) n.textContent = text;
  return n;
}
function chart(id, points, key, label, percent = false) {
  const valid = points.filter(p => p[key] != null && Number.isFinite(p[key]));
  if (!valid.length) { el(id).replaceChildren(node("p", "Waiting for measurements…", "muted")); return; }
  const svg = svgNode("svg", {viewBox: "0 0 800 190", role: "img", "aria-label": label});
  const xValue = p => percent ? p.updates : p.timestamp;
  const xMin = xValue(valid[0]), xMax = xValue(valid[valid.length - 1]);
  const yMax = percent ? 1 : Math.max(0.01, ...valid.map(p => p[key])) * 1.1;
  const x = p => 45 + 730 * (xValue(p) - xMin) / (xMax - xMin || 1);
  const y = p => 145 - 125 * p[key] / yMax;
  for (let i = 0; i <= 4; i++) {
    const yPos = 145 - i * 125 / 4;
    svg.append(svgNode("line", {x1:45, x2:775, y1:yPos, y2:yPos, class:"grid"}));
    svg.append(svgNode("text", {x:36, y:yPos+4, "text-anchor":"end"}, percent ? `${i*25}%` : (yMax*i/4).toFixed(2)));
  }
  svg.append(svgNode("polyline", {points:valid.map(p => `${x(p)},${y(p)}`).join(" "), class:"line"}));
  if (percent) for (const p of valid) {
    const circle = svgNode("circle", {cx:x(p), cy:y(p), r:p.promoted ? 5 : 3, class:p.promoted ? "promoted" : "point"});
    circle.append(svgNode("title", {}, `Update ${p.updates}: ${pct(p.score)}${p.promoted ? " · published / baseline" : " · not promoted"}`));
    svg.append(circle);
  }
  svg.append(svgNode("text", {x:45,y:164}, percent ? fmt(xMin) : time(xMin)));
  svg.append(svgNode("text", {x:775,y:164,"text-anchor":"end"}, percent ? fmt(xMax) : time(xMax)));
  svg.append(svgNode("text", {x:410,y:185,"text-anchor":"middle"}, label));
  el(id).replaceChildren(svg);
}
function render(s) {
  state = s;
  const b = s.background, c = b.candidate || {}, p = s.published, config = b.config;
  const known = s.models.join("|");
  if (el("model").dataset.known !== known) {
    el("model").replaceChildren(...s.models.map(name => { const option = node("option", name); option.value=name; return option; }));
    el("model").dataset.known = known;
  }
  el("model").value = s.model_id;
  const heartbeatAge = Date.now()/1000 - b.heartbeat;
  el("connection").textContent = b.running ? "API connected" : "API connected · learner stopped";
  el("connection").className = `badge ${b.running ? "live" : "offline"}`;
  const warnings = [];
  if (b.error) warnings.push(`Learner stopped: ${b.error}`);
  if (b.learner_checkpoint.error) warnings.push(`Learner checkpoint: ${b.learner_checkpoint.error}`);
  if (p.checkpoint_error) warnings.push(`Published checkpoint: ${p.checkpoint_error}`);
  if (heartbeatAge > 60 && b.running) warnings.push(`No worker heartbeat for ${Math.round(heartbeatAge)}s; the current search, evaluation or save may be slow.`);
  el("alert").hidden = !warnings.length; el("alert").textContent = warnings.join(" ");
  el("games").textContent = fmt(b.games);
  el("throughput").textContent = `${fmt(b.games_per_second)} games/s · ${fmt(b.observations_per_second)} positions/s · last completed batch`;
  el("candidate-update").textContent = `${fmt(c.updates)} updates`;
  el("candidate-detail").textContent = `${fmt(c.observations)} observations accumulated · ${fmt(b.replay_size)} simulated positions in replay`;
  const latest = b.last_evaluation;
  if (el("overview-status")) {
    const tasks={"collecting experience":"Practicing",optimizing:"Learning from recent games","baseline evaluation":"Taking its first test","promotion evaluation":"Checking for improvement","saving learner":"Saving progress",paused:"Practice paused",stopped:"Practice stopped",error:"Needs attention",starting:"Starting practice"};
    el("overview-status").textContent=tasks[b.phase]||"Practicing";
    el("overview-summary").textContent=b.error?"Practice has stopped. Open More details to check the error.":"One AI practices in the background. You play the latest version that passed a test.";
    const test=s.evaluations[s.evaluations.length-1];
    el("overview-score").textContent=test?pct(test.score):"Not tested yet";
    el("overview-score-note").textContent=test?`${test.wins} wins, ${test.draws} draws, ${test.losses} losses in the latest practice-version test.`:"Its first test will appear here.";
    el("overview-version").textContent="The version visitors play. It may differ from the practice copy.";
  }
  el("gate").textContent = latest?.arena ? pct(latest.arena.score) : "Awaiting check";
  el("gate-detail").textContent = latest?.arena ? `${latest.promoted ? "Published" : "Kept training"} · ${latest.arena.wins}W / ${latest.arena.draws}D / ${latest.arena.losses}L against published model` : `First candidate check after ${fmt(config.evaluation_seconds)}s and replay warmup.`;
  el("generation").textContent = `${el("overview-status")?"Version":"Generation"} ${p.generation}`;
  el("published-detail").textContent = `Weights from update ${fmt(p.weights_update)} · ${p.published_at ? `published ${age(p.published_at)}` : "initial search-enabled network"}`;
  el("phase").textContent = b.phase;
  el("phase-detail").textContent = `${age(b.phase_since)} · heartbeat ${age(b.heartbeat)}`;
  el("pause").textContent = b.pause_requested ? "Resume practice" : "Pause practice";
  for (const id of ["pause", "evaluate", "checkpoint"]) el(id).disabled = commanding || !b.running;
  el("device").textContent = `${s.device.toUpperCase()} · ${fmt(s.parameters)} parameters`;
  const frame = b.latest_game;
  board("sample-board", frame?.board);
  el("sample-age").textContent = age(frame?.updated_at);
  el("sample-detail").textContent = frame ? `${frame.opponent === "self" ? "Recent published opponent" : frame.opponent} · ${frame.move_count} discs · ${frame.winner == null ? `next player ${frame.next_player}` : frame.winner === 0 ? "draw" : `player ${frame.winner} won`}. Positions are sampled across games.` : "Waiting for a training position.";
  facts("replay", [["Simulated replay", `${fmt(b.replay_size)} / 100,000`], ["External replay", `${fmt(b.human_replay_size)} / 10,000`], ["External episodes queued / consumed", `${fmt(b.external_received)} / ${fmt(b.external_consumed)}`], ["Positions from external games this process", fmt(b.external_positions)], ["Waiting to enter replay", `${b.queue_size} episodes`]]);
  rows("mixture", Object.entries(b.opponents).map(([kind, count]) => [kind === "self" ? "Recent AI" : kind, pct(config.mixture[kind]), fmt(count.games), `${count.wins} / ${count.draws} / ${count.losses}`]), "No games yet", 4);
  facts("losses", [["Combined policy + value loss", fmt(c.loss)], ["Policy loss", fmt(c.policy_loss)], ["Value loss", fmt(c.value_loss)], ["Policy entropy (nats)", fmt(c.entropy)], ["Gradient norm before clipping at 5", fmt(c.gradient_norm)], ["Search budget · training / visitor", `${config.training_simulations} / ${config.serving_simulations}`], ["Concurrent practice games", config.batch_games]]);
  chart("loss-chart", b.series, "loss", "Combined loss · wall clock");
  chart("strength-chart", s.evaluations, "score", "Learning updates · bright dots mark released versions or the starting AI", true);
  el("next-check").textContent = b.evaluation_requested ? "Extra check requested" : b.phase.includes("evaluation") ? "Evaluation in progress" : b.pause_requested ? "Learning paused" : `Next check in ~${Math.max(0, Math.ceil(b.evaluation_due_at-Date.now()/1000))}s`;
  if(el("overview-status"))el("next-check").textContent=b.evaluation_requested?"Progress check requested":b.phase.includes("evaluation")?"Test in progress":b.pause_requested?"You can still play":"Next test in about "+Math.max(1,Math.ceil((b.evaluation_due_at-Date.now()/1000)/60))+" min";
  rows("evaluations", [...s.evaluations].reverse().map(point => [time(point.timestamp), fmt(point.updates), `${pct(point.score)} · ${point.wins}/${point.draws}/${point.losses}`, point.random ? pct(point.random.score) : "—", point.arena ? `${pct(point.arena.score)} · ${point.arena.games} games` : "—", point.positions ? `${point.positions.first.wins}/${point.positions.first.draws}/${point.positions.first.losses} · ${point.positions.second.wins}/${point.positions.second.draws}/${point.positions.second.losses}` : "—", [point.stage === "baseline" ? "Baseline" : point.promoted ? "Published" : "Not promoted", point.promoted ? "good" : "warn"]]), "First evaluation is running in the background.", 7);
  el("queue").textContent = `${b.queue_size} queued episodes`;
  facts("persistence", [["Learner checkpoint", b.learner_checkpoint.path], ["Learner last saved", age(b.learner_checkpoint.saved_at)], ["Learner save pending", b.checkpoint_requested ? "Yes" : "No"], ["Published checkpoint", p.checkpoint_path], ["Published last saved", age(p.checkpoint_saved_at)], ["Worker uptime", `${Math.round((Date.now()/1000-b.started_at)/60)} min`]]);
  const requests = s.requests.filter(r => r.path !== "/debug/analyze");
  const latencies = requests.map(r => r.milliseconds).sort((a,b) => a-b);
  const quantile = q => latencies.length ? `${fmt(latencies[Math.min(latencies.length-1, Math.floor(q*latencies.length))])} ms` : "—";
  facts("request-metrics", [["Retained request measurements", `${requests.length} (last 100 total)`], ["Median / 95th percentile", `${quantile(0.5)} / ${quantile(0.95)}`], ["Recent rejected game requests", requests.filter(r => r.status >= 400).length]]);
  facts("errors", [...Object.entries(s.errors).map(([kind, count]) => [kind, count, count ? "bad" : ""]), ["Last service error", s.last_error || "None"]]);
  el("session-count").textContent = `${s.sessions.length} retained`;
  rows("sessions", s.sessions.map(session => [session.game_id, Object.values(session.bindings).some(binding => binding.train) ? "External learning" : "Exhibition / inference", `${Math.round(session.age_seconds)}s`, session.moves, session.pending_players.join(", ") || "—", session.recorded_positions, session.finished ? "Finished · awaiting deletion" : "In progress"]), "No active sessions for this model.", 7);
  el("events").replaceChildren(...[...b.events].reverse().map(event => {
    const row = node("div", null, "event"); row.append(node("time", time(event.timestamp)), node("b", event.kind, event.kind === "error" ? "bad" : event.kind === "publication" ? "good" : ""));
    const message = node("div", event.message);
    const details = Object.entries(event).filter(([key]) => !["timestamp", "kind", "message"].includes(key)).map(([key, value]) => `${key}: ${typeof value === "number" ? fmt(value) : value}`).join(" · ");
    if (details) message.append(node("small", details)); row.append(message); return row;
  }));
  if (!b.events.length) el("events").append(node("p", "Waiting for events. Earlier process history remains in checkpoint evaluation history.", "muted"));
  el("freshness").textContent = `Snapshot ${new Date(s.timestamp*1000).toLocaleString()} · refreshes every 2s · ${s.architecture} · metrics are scoped to ${s.model_id}`;
}
async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {cache:"no-store"} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail || result));
  return result;
}
async function poll() {
  const requestedModel = selectedModel;
  try {
    const snapshot = await request(`/debug/data?model=${encodeURIComponent(requestedModel)}`);
    if (requestedModel === selectedModel) render(snapshot);
  } catch (error) {
    el("connection").textContent = "Disconnected"; el("connection").className = "badge offline";
    el("alert").hidden = false; el("alert").textContent = `Debug data unavailable: ${error.message}. Retrying automatically.`;
  } finally { setTimeout(poll, document.hidden ? 5000 : 2000); }
}
el("model").addEventListener("change", () => { selectedModel=el("model").value; state=null; el("analysis").replaceChildren(); });
async function control(action) {
  if (commanding) return;
  commanding = true;
  for (const id of ["pause", "evaluate", "checkpoint"]) el(id).disabled=true;
  try {
    const result = await request("/debug/control", {model_id:selectedModel, action});
    const messages={pause:"Practice will pause after the current games. You can still play.",resume:"Practice will resume shortly.",evaluate:"Progress check requested. Results appear when the test finishes. Resume practice first if it is paused.",checkpoint:"Progress will be saved after the current practice games."};
    el("command-status").textContent = messages[action]||result.note;
  } catch (error) { el("command-status").textContent = error.message; }
  finally { commanding=false; }
}
el("pause").addEventListener("click", () => control(state?.background.pause_requested ? "resume" : "pause"));
el("evaluate").addEventListener("click", () => control("evaluate"));
el("checkpoint").addEventListener("click", () => control("checkpoint"));
function setPosition(value) {
  el("inspect-board").value = JSON.stringify(value);
  const count = value.flat().reduce((sum, disc) => sum+disc, 0);
  el("inspect-player").value = count === 0 ? "1" : "-1";
  board("inspect-preview", value);
  el("analysis").replaceChildren();
  el("analysis-status").textContent = "Position copied. Analyze it to compare network preferences and search.";
}
el("inspect-empty").addEventListener("click", () => setPosition(empty()));
el("inspect-sample").addEventListener("click", () => { if (state?.background.latest_game) setPosition(state.background.latest_game.board); });
el("inspect-exhibition").addEventListener("click", async () => {
  try { setPosition((await request("/state")).board); } catch(error) { el("analysis-status").textContent=error.message; }
});
el("inspect-board").addEventListener("change", () => { try { board("inspect-preview", JSON.parse(el("inspect-board").value)); } catch { el("analysis-status").textContent="Board JSON is invalid."; } });
el("analyze").addEventListener("click", async () => {
  el("analyze").disabled=true; el("analysis-status").textContent="Inspecting a copy of published weights…";
  try {
    const data = JSON.parse(el("inspect-board").value);
    const result = await request("/debug/analyze", {model_id:selectedModel, board:data, player:Number(el("inspect-player").value), simulations:Number(el("inspect-budget").value)});
    board("inspect-preview", result.board);
    rows("analysis", result.columns.map(c => [[c.column, c.column === result.column ? "good" : ""], c.legal ? "Yes" : "No", pct(c.prior), `${c.visits} / ${result.simulations}`, c.value == null ? "Unvisited" : c.value.toFixed(3)]), "", 5);
    el("analysis-status").textContent = `Generation ${result.generation}, weights from update ${fmt(result.weights_update)} · chooses column ${result.column} · network value ${result.network_value.toFixed(3)} · search value ${result.search_value.toFixed(3)} · ${fmt(result.milliseconds)} ms. No learning or game action performed.`;
  } catch(error) { el("analysis-status").textContent=error.message; }
  finally { el("analyze").disabled=false; }
});
el("download").addEventListener("click", () => {
  if (!state) return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:"application/json"}));
  const link=node("a"); link.href=url; link.download=`connect4-debug-${selectedModel}-${Date.now()}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
setPosition(empty()); board("sample-board",empty()); poll();
