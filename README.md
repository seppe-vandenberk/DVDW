# Connect Four learning lab

A STEM-day demonstration of learning through repetition and rewards, on a **7-column × 5-row** gravity board. The default engine is a small policy/value network guided by Monte Carlo tree search. It starts with random weights, learns from completed games, and publishes evaluated snapshots for visitors.

Background learning continues during browser and physical human games. The home page is a focused Play screen using the same design and navigation as the operator console. Training observation, candidate benchmark scores, publication results, and a sampled live background-training board live in the debug console.

## Install

The repository has two parts with **separate environments**; install only what you need.

| Part | Needs | Install |
|---|---|---|
| Model API + viewer (root, `connect4/`) | Docker, **or** Python 3.12 + `pip install -r requirements.txt` in `.venv` | `docker compose up --build -d`, or see [Run](#run) |
| Camera reader (`camera/`, optional, needs an NVIDIA GPU) | conda env with Python 3.12, PyTorch + CUDA, SAM 3, access to `facebook/sam3` on Hugging Face | [camera/README.md](camera/README.md#2-install-on-linux) (Linux) or [Windows](camera/README.md#3-install-on-windows) |

```powershell
git clone https://github.com/seppe-vandenberk/DVDW.git
cd DVDW
python -m venv .venv                                   # model API + viewer without Docker
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The camera reader pins `numpy<2` and its own dependencies, so keep it in its own conda environment (`camera/requirements.txt`) and do not install it into `.venv`. Without a camera you can skip it completely and use the mock camera. Running everything together is described in [Run the whole system](#run-the-whole-system).

## Run

```powershell
docker compose up --build -d
```

- Viewer: http://localhost:8080
- STEM participant display: http://localhost:8080/participant
- STEM instructor display: http://localhost:8080/instructor
- Model API / documentation: http://localhost:8000/docs
- Camera reader state (only when `camera/` is running): http://localhost:8001/state
- Worker status: http://localhost:8000/health
- Quick overview (advanced tools under More details): http://localhost:8080/debug
- Illustrated model explanation: http://localhost:8080/explain

CPU is the default; a GPU is optional. For NVIDIA Docker support:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

Use the same Compose files for later commands. Benchmark CPU versus CUDA for this small network. Stop an older local API/simulator before starting Compose; ports and the model directory are shared. Use one API process/worker and one instance per model directory.

Without Docker, install dependencies and start the API:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:AI_BACKEND = "az"
$env:DQN_DEVICE = "cpu"
.\.venv\Scripts\python -m uvicorn connect4.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal:

```powershell
.\.venv\Scripts\python -m uvicorn connect4.simulation_service:app --host 127.0.0.1 --port 8080 --workers 1
```

The API starts local training automatically; opening the viewer is optional. The legacy HTTP exhibition loop is disabled by default. Opening Play does not start or reset a game: press Start game to begin, or resume the existing shared human game. Background learning continues throughout. Stop the API gracefully to save training and drain confirmed human outcomes.

## Learning architecture

- Local bitboard games avoid per-move HTTP calls. Neural inference is batched across independent search trees. One background thread per loaded model owns its learner.
- The network predicts seven move preferences and a position value in `[-1, 1]`. Inputs encode the player about to move as `+1`.
- PUCT search uses legal game transitions and learned estimates, backing up values with alternating signs. There are no hardcoded win/block overrides or rule-teacher labels for the learner.
- Opponents are initially mixed 60% recent frozen published networks, 25% rule player, and 15% random. Learner color alternates. The historical pool holds four published snapshots; after restart it begins with the current published network.
- Training searches add root noise and sample visit counts during the first ten plies. Later training moves and visitor moves choose the most visited action. Viewer configuration controls exhibitions, independently of the background mixture.
- Completed games supply search distributions as policy targets and outcomes as value targets: win `+1`, draw `0`, loss `−1`, discounted by `0.997` per ply. This prefers earlier wins and later losses, without extra pair/center rewards.
- Replay holds 100,000 simulated positions and 10,000 external positions. When available, up to 20% of minibatches is reserved for external experience. Horizontal reflection mirrors both board and policy.
- Human actions are opponent replies, not expert move labels. Confirmed API games supply the AI's search policy and actual outcome. Aborted games receive no invented result. HTTP exhibitions are inference-only and do not fill human replay.
- Serving and learning use separate networks. Training, evaluation, and learner checkpoint serialization run outside the API lock; publication briefly locks to replace serving weights. They share one process, so compute contention remains possible.

## Evaluation and publication

The debug console separates candidate updates from published-weight updates, shows current worker phase, rates, losses, opponent results, replay sources, queued external episodes, checkpoint ages/errors, active game sessions, request timings, and recent events. Pause/resume affects background learning only; evaluation requests wait while paused. Save requests are handled between batches. The move inspector analyzes a copy of the published network without changing games, experience, exploration counters, or weights, and compares neural priors with search visits/values. Export snapshot downloads the current telemetry as JSON.

The explanation page gives a short, plain-language introduction: practice, learn patterns, look ahead, then test. Three SVG diagrams illustrate those ideas. Optional details include an interactive reward example and a link to the full technical explanation with architecture, board encoding, search, publication, and physical-game diagrams. The overview shows practice games, latest practice-test score, and playable version first; detailed diagnostics remain collapsed under More details.

The worker measures the initial published network against rules. About every five minutes it checks the candidate against the published network in 40 games and against rules/random play in 20 games each. Colors are balanced; paired games share two seeded opening plies. Evaluation disables exploration. Rule/random opening seeds stay fixed, while promotion seeds vary with optimizer updates.

A candidate scoring at least 60% against the published model is published. This practical gate is **not a statistical guarantee of improvement against every opponent**. The chart includes candidates that were not published and identifies them. Checks use the training search budget, potentially different from physical play.

`POST /evaluate` queues a background check and returns immediately. `GET /models/shared/progress` reads results without running evaluation. Games, observations, optimizer updates, loss, and playing strength are different metrics.

Checkpoints are separate from preserved DQN weights:

- `models/az-shared.pt`: published serving weights and progress.
- `models/az-shared.learner.pt`: candidate, optimizer, replay, and progress.

Learner saves are atomic, about every minute and at shutdown. Serving snapshots save on publication and shutdown. `POST /checkpoint` saves serving weights and queues a learner save; the response reports a pending save. Active physical sessions and unconfirmed moves are not persisted. A crash can lose recent unsaved experience. After restart use new game IDs.

For a fresh event run, choose a new `MODEL_DIR` or an empty model volume. Do not delete checkpoints while their API is running. Old `models/shared.pt` weights are preserved and never loaded into the new architecture.

## Configuration

Copy `.env.example` to `.env` for Compose. Direct Python runs require terminal environment variables; Python does not load `.env` automatically.

| Variable | Default | Meaning |
|---|---:|---|
| `AI_BACKEND` | `az` | New engine; `dqn` opts into legacy |
| `AI_BACKGROUND` | `true` | Enable local learning; `false` does not consume external queues |
| `AI_SEARCH_SIMULATIONS` | `64` | Physical/exhibition search budget |
| `AI_TRAIN_SIMULATIONS` | `32` | Background/evaluation search budget |
| `AI_BATCH_GAMES` | `16` | Local games per batch |
| `AI_EVALUATION_SECONDS` | `300` | Minimum time between candidate checks |
| `AI_GATE_GAMES` | `40` | Balanced promotion-match game count |
| `DQN_DEVICE` | `cpu` in Compose | Device for either engine |
| `SIM_INTERVAL` | `0.25` in Compose | Exhibition pacing; does not throttle learning |

Use the single `shared` model for the event, so all visitors contribute to one learner. At most four models can be loaded, each with a worker. PyTorch uses one intra-operation CPU thread. Legacy exploration-reset controls are hidden for the new backend; it uses search noise and early-game sampling instead.

## Rehearse the event

```powershell
.\.venv\Scripts\python -m connect4.rehearse --directory runtime/rehearsal-five-hours --seconds 18000 --seeds 2026 43 29
```

This runs **five hours per seed, sequentially**, about fifteen hours total plus final evaluation. For one five-hour run use `--seeds 2026`. Output directories must be new; live models are untouched. Each seed saves checkpoints and `summary.json`, including throughput and separate final evaluations of published/candidate models against random, rules, and a depth-four alpha-beta opponent. Final opening seeds differ from promotion checks. The fixed search opponent has handcrafted line scores; it is not a perfect solver.

Quick pipeline rehearsal:

```powershell
.\.venv\Scripts\python -m connect4.rehearse --directory runtime/quick-check --seconds 120 --seeds 2026 --test-games 20 --evaluation-seconds 45 --gate-games 20
```

A local CPU smoke run completed 2,016 games in about 120 seconds, collected about 16,000 observations, and published twice. On 20 held-out games per opponent, the published model scored 100% against random, 37.5% against rules, and 30% against depth-four search. This validates the pipeline and early progress, **not five-hour mastery**. Its local artifact is `runtime/az-smoke-20261111/2026/summary.json`.

Literal optimal play requires an exact solver for this 7×5 variant or a separate proof. Search already contributes strength before learning: show the initial search-enabled benchmark so visitors can distinguish learning gains from look-ahead ability.

## Physical camera / robot API

Board format is `board[column][row]`, row 0 at the top, seven columns of five rows. Empty = `0`, first player = `1`, second player = `-1`; labels remain fixed throughout the game. Transpose row-major camera arrays.

On the robot's turn:

```json
POST /move
{
  "game_id": "physical-001",
  "board": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
  "player": 1,
  "model_id": "shared",
  "train": true,
  "explore": false
}
```

The response includes `column`, `predicted_board`, `done`, `winner`, `updates`, `replay_size`, and diagnostics. `done` describes what happens **if the requested disc is successfully placed**. Place exactly one disc, confirm it with the camera, wait for the human reply, then send the next board with the same game ID. The API accepts its predicted placement plus at most one opponent move. Identical requests are cached; the robot must also independently prevent duplicate placements on retries.

After **either player wins or the board fills**, send `POST /finish` with `game_id` and the actual final `board`, even if `/move` predicted a win. Confirmed results queue one episode per participating learned player. Then call `DELETE /games/physical-001`. For interruptions or camera/placement errors, delete the unfinished session and start with a new ID and corrected board. Do not reuse deleted IDs. Keep model bindings and training settings fixed within each game.

Use `train:false` for inference-only games. At most 1,000 sessions are retained. Board validation errors return 422; turn, confirmation, and session conflicts return 409. The server has no authentication and is intended for a trusted event LAN.

## Tests and legacy code

```powershell
.\.venv\Scripts\python -m pytest -q
```

The existing regression tests explicitly select legacy DQN through their fixture. New tests select `backend="az"` and verify bitboards, search signs and tactics, training/checkpoint restoration, camera retries, outcome queues, balanced benchmarks, worker lifecycle, and publication behavior.

The previous implementation remains available for comparison with `AI_BACKEND=dqn`; its original documentation is [LEGACY_DQN.md](LEGACY_DQN.md).

## STEM-day physical displays

Open `/participant` on the participant screen and `/instructor` on the instructor screen. Both are full-viewport displays without navigation or page scrolling. Use a desktop/laptop screen for the complete instructor dashboard. Browser full-screen mode (F11) also hides browser chrome.

The instructor selects a robot character, a human character, the robot's blue (`B`) or red (`R`) discs, and who starts. The other color is assigned to the human. Start new game applies these selections and clears the shared event game. Character selections are decorative; player numbers are assigned by turn order for the existing model API. The original browser Play game remains at `/`.

One server-owned loop reads the camera every two seconds, independently of open browser tabs. Blank `CAMERA_API_URL` selects the shared mock camera, exposed at `GET /event/camera/mock`. The instructor can load/edit/send its response. This is row-major: five rows of seven `X`, `B`, or `R` cells, plus `game_id`, an ISO-8601 `timestamp`, and `valid`. The mock endpoint refreshes the timestamp when read; the editor refreshes it when sending. Set `CAMERA_API_URL` to the camera reader's endpoint when ready (see [Camera reader](#camera-reader-camera) below) (also passed through Docker Compose). External timestamps must be within ten seconds of server time; invalid, stale, floating, and impossible boards are rejected and reported.

A changed, valid board starts a model request only on the robot's turn. Timestamps alone never trigger another request or restart its animation. The robot thinks for a minimum of one second, rising linearly to three seconds as the board fills, and longer if inference is still running. It carries a disc to the selected slot, then lights that slot. This is a placement instruction: a suggested disc is never silently added to the observed board, and no hardware command is sent. Confirm the robot disc and human reply through camera observations.

The instructor sees camera and model connection status, last successful response times, latency, and errors. Retry move API explicitly bypasses unchanged-board suppression, while in-flight requests remain protected. Manual controls place discs for either player with gravity and turn validation, then pause camera updates. Resume camera starts a fresh model session and reads the physical board again. If a manual robot move differs from the suggestion, the system starts a new session without assigning an invented training result. Terminal observations call `/finish` before deleting the completed model session. Event display state is in memory and resets on service restart; run one viewer process for a shared event.

Robot assets are bundled at `connect4/static/robot/waiting.gif`, `thinking.gif`, and `still.png`. They are original programmatically drawn looping sprites; browser motion carries the disc to the selected column. Reduced-motion settings use the still image and suppress travel/pulsing.

Focused checks:

```powershell
.\.venv\Scripts\python -m pytest tests/test_event_game.py tests/test_simulation_config.py -q
node tests/event_ui_smoke.cjs
```

The participant screen shows Dutch turn labels, player color swatches, and a cumulative robot score chart: robot win +1, loss -1, draw 0. Each observed terminal game counts once. Scores survive New game and service restarts through `runtime/robot_scores.txt` (`EVENT_SCORE_FILE` overrides this for direct runs). Compose stores the file at `runtime/event/robot_scores.txt` on the host; the graph displays the most recent 30 games.

Scores are plain text: one `+1`, `-1`, or `0` per game. Saves replace the file atomically and restore the cumulative graph at startup. Instructor end-screen previews (AI verliest / AI wint / Stop test) are shared with the participant screen and never change the board or scores.

Instructor **Bordbreedte aanpassen** opens the width editor (50-175%, smaller/larger and reset). It updates the participant board, slots, and robot rail together, and survives New game within the running service. Confirmed human and robot placements, including manual controls, animate falling discs; unchanged polling snapshots do not replay them. Reduced-motion preferences suppress this animation.
"# DVDW" 


## Camera reader (`camera/`)

[camera/](camera/) is the former `dvdw-connect4` project (history preserved). It reads the physical board with a camera or video using SAM 3 and serves it at `GET /state`. That is exactly what `CAMERA_API_URL` consumes: `board` (5 rows x 7 `X`/`R`/`B`, row 0 on top), `game_id`, `timestamp`, `valid`. It needs an NVIDIA GPU and runs **on the host, not in Docker**. Installation, calibration and all options are in [camera/README.md](camera/README.md).

```
camera/video -> camera/connect4_reader.py (:8001 /state) -> simulator (:8080, polls CAMERA_API_URL) -> model API (:8000 /move)
```

| Component | Folder | Port | Runs in |
|---|---|---|---|
| Model API (learning, `/move`) | `connect4/api.py` | 8000 | Docker or host |
| Viewer / STEM displays | `connect4/simulation_service.py` | 8080 | Docker or host |
| Camera reader | `camera/connect4_reader.py` | 8001 | host (GPU) |

### Run the whole system

**One-time setup:** install the camera reader following [camera/README.md](camera/README.md) (conda env, PyTorch + CUDA, SAM 3, `pip install -r camera/requirements.txt`, Hugging Face login). Docker is only needed for the model API and viewer.

**1. Start the camera reader** (own terminal, conda env active). The event board has 5 rows, so `--rows 5` is required:

```powershell
python camera/connect4_reader.py --rows 5 --port 8001                 # RealSense camera (calibrate on first run)
python camera/connect4_reader.py --rows 5 --port 8001 --camera 0      # webcam
python camera/connect4_reader.py --rows 5 --port 8001 --video game.mp4   # recorded video
```

Check it with `Invoke-RestMethod http://localhost:8001/state` (`valid` stays `false` until SAM 3 has loaded).

**2. Point the viewer at it.** Copy `.env.example` to `.env` and set:

```
CAMERA_API_URL=http://host.docker.internal:8001/state
```

**3. Start the model API and viewer:**

```powershell
docker compose up --build -d            # add  -f compose.gpu.yaml  for NVIDIA GPU training
```

**4. Open the displays:** instructor http://localhost:8080/instructor (should say "External camera" and show the camera as connected), participant http://localhost:8080/participant.

**Without Docker** (three terminals, from this folder):

```powershell
# terminal 1: model API
$env:AI_BACKEND = "az"; $env:DQN_DEVICE = "cpu"
.\.venv\Scripts\python -m uvicorn connect4.api:app --host 127.0.0.1 --port 8000 --workers 1
# terminal 2: viewer, using the camera reader
$env:CAMERA_API_URL = "http://127.0.0.1:8001/state"
.\.venv\Scripts\python -m uvicorn connect4.simulation_service:app --host 127.0.0.1 --port 8080 --workers 1
# terminal 3: camera reader (conda env)
python camera/connect4_reader.py --rows 5 --port 8001
```

**Without a camera:** leave `CAMERA_API_URL` blank. The viewer then uses the built-in mock camera that the instructor page can edit.

**Stop:** `docker compose down`, then Ctrl+C (or `q`/`Esc` in the preview window) for the camera reader.

### Troubleshooting the connection

- *Camera shows as unavailable on the instructor page:* open the `/state` URL in a browser; `valid` must be `true`. An invalid board (floating discs, wrong disc counts), a stale timestamp (clocks more than 10 s apart) or a board that is not 5x7 is rejected, and the reason is shown next to the camera status.
- *Board has 6 rows:* the reader was started without `--rows 5`.
- *Port 8000 already in use:* the camera reader must use 8001 (its default); 8000 belongs to the model API.
- *Docker cannot reach the reader:* use `host.docker.internal` (configured in `compose.yaml`), and allow Python through the Windows firewall for port 8001.
- Calibrate again whenever the camera or the board is moved (`--recalibrate`).
