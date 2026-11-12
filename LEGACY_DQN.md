# Legacy Connect Four DQN (opt-in with AI_BACKEND=dqn)

A 5,112-parameter PyTorch CNN with CPU/CUDA support, a FastAPI model server, and a separate HTTP training simulator for a **7-column by 5-row** gravity board. Actions are columns `0..6`; four connected disks win. By default, the learning AI plays against a fixed rule-based opponent.

## Run

### Docker (model API + live simulator viewer)

With Docker Desktop running in **Linux-container / WSL 2 mode** and NVIDIA GPU support:

```powershell
docker compose up --build -d
```

- Viewer: http://localhost:8080 — the actual simulator board, next player, last column, game results, exploration, training loss, replay, updates, and device. No browser installation, Node, or frontend build is needed.
- Model API / camera integration: http://localhost:8000/docs
- Health: http://localhost:8000/health and http://localhost:8080/health

The API container owns the model and checkpoints. CPU is the default for this tiny network; set `DQN_DEVICE=cuda` in `.env` to use the included CUDA runtime. The lightweight simulator container runs AI-versus-rules games and serves the viewer, polling its own state every 250 ms. The AI and rule-based opponent alternate colors, so each gets to start. The pause between games is 0.0001 seconds. It reconnects automatically if the API is restarted.

Use **Simulation configuration** to choose a learning AI or the rule-based opponent for each side. The first selected player is yellow and starts. Choose **Learning AI** for both sides for AI self-play; use the same model name (such as `shared`) to train shared weights, or different names for separate learned models. At least one side must be a learning AI. **Swap players after each game** alternates their colors and starting order when enabled, including two distinct AIs; disabling it keeps the selected positions fixed.

**Apply configuration** cancels the unfinished game without assigning an invented outcome, keeps all learned weights and experience, and starts a fresh simulation in the exact order selected. It also resumes simulation if human mode was active. Switching starts from the selected order each time a configuration is applied, and human games do not advance its alternating phase. Settings are shared by all viewer tabs for the running service; on service restart, environment defaults are restored. Human mode uses the selected learned participant (the second AI when both sides are learned models).

The rule-based opponent chooses its first move uniformly at random from legal columns in every game, whether it plays first or second. After that, it follows this priority: finish four in a row; block an immediate opposing win; avoid moves that give the opponent an immediate winning reply; block an opposing pair when its own usable lines are only singles; otherwise extend its strongest usable sequence, then prefer promising lines and the center. Sequences blocked from ever making four are ignored. Only legal gravity moves count as immediate threats. If no safe move exists, it minimizes the opponent's immediate winning replies. Equal choices are randomized. The viewer explains the rule used for the last move.

Click **Play as human** to pause automatic simulation and start a game against the learned AI (`shared` by default). You play yellow (player 1) and move first; click a column number or a board cell to drop a disc. The red AI chooses its best legal move without exploration, and still trains from your replies and the confirmed outcome. **New game** starts another human game; **Watch simulation** resumes automatic training with a fresh game. Switching modes discards the unfinished game without inventing a win or loss. These controls affect the shared viewer service, including other open browser tabs. A failed API action can be retried with **Retry bot action**; the same human disc is not placed twice.

The **Learning progress** panel plots benchmark score against optimizer updates for each configured learned model. A check plays 50 games against the same rule-based opponent used in simulation. The bot plays 25 games as each color, using greedy moves without learning or consuming exploration decisions. Score is `(wins + 0.5 * draws) / 50`; the panel also shows win rate and average own discs per win. The same seeds select random openings and break tied rule choices, so later checkpoints are comparable. This is a small benchmark against one opponent, not a general playing-strength rating; scores can go down as well as up. Results from older benchmark versions are excluded from the chart.

Opening the panel establishes a starting measurement. Once established, the API measures automatically every 250 optimizer updates, including training from human games. **Measure now** runs an extra check. Checks briefly hold the model lock so their weights stay fixed. The latest 100 measurements are saved in each model checkpoint and survive restarts. Progress refreshes every five seconds; unchanged polls do not rewrite checkpoints.

The **Reset exploration** buttons set exploration to **25%, 50%, 75%, or 100%** for every distinct learned model configured in the viewer. Each button starts a new decay schedule from the selected rate and resets the displayed decisions-since-reset count to zero. It preserves learned weights, optimizer updates, replay memory, active games, and benchmark history, and saves immediately. Exploration decreases by 0.002 percentage points per exploratory decision toward the 5% minimum; human/camera games still play greedily. The starting rate and decisions counter persist across API restarts. For direct API clients, use `POST /exploration/reset` with `{"model_id":"shared","rate":0.25}` (omitting `rate` still means 100%), `GET /models/shared/progress`, or `POST /evaluate` with the model payload. The viewer's reset endpoint accepts `{"rate":0.25}` and applies it to its configured learned models. Restart/rebuild both services after installing these changes.

`./models` is mounted into the API container, so your existing `shared.pt` continues training and survives image rebuilds or `docker compose down`. **Stop any locally running API/simulator before starting Compose:** the API port is shared, and two servers must never write the same checkpoint files. Never run this Compose project twice against the same `models` folder.

For a complete fresh start, a `models/<model_id>.reset` marker instructs the updated API to delete that model's old checkpoint on its next load, create fresh networks and optimizer, and save empty replay/history with exploration back at 100%. The marker is removed after the fresh checkpoint is saved. Rebuild/restart the API and simulator with `docker compose up --build -d` to consume a reset or staged checkpoint. A staged `<model_id>.next.pt` takes precedence over an older reset marker. This deferred handoff prevents an older running API from restoring its old weights after an external checkpoint deletion. Ordinary later restarts resume the new training.

```powershell
docker compose logs -f simulator
docker compose logs -f api
docker compose stop
docker compose start
docker compose down
```

`stop`/`down` let the simulator clean up its current game and the API save before exit. `start` resumes existing containers; `up --build -d` rebuilds when code changes. Restart policies keep both services running while Docker is active. Sleep or shutdown pauses training.

To customize startup defaults, copy `.env.example` to `.env` and edit its values, then run `docker compose up -d`. `SIM_INTERVAL=0` removes the delay between moves; `SIM_RESULT_DELAY=0.0001` sets the pause after each result. Defaults are `MODEL_ONE=shared`, `MODEL_TWO=rules`, and `SIM_SWAP_SIDES=true`. Use `SIM_SWAP_SIDES=false` to start with fixed player positions. Player choices and switching can then be changed live in the viewer. `rules` is reserved for the fixed opponent and has no checkpoint, exploration schedule, or training requests. `BIND_HOST=0.0.0.0` makes the camera API and viewer reachable on your trusted LAN. The default only exposes localhost. Session scores reset when the simulator restarts; model learning persists.

CUDA in Docker Desktop on Windows requires its [WSL 2 GPU backend](https://docs.docker.com/desktop/features/gpu/). The Compose file [reserves GPU access](https://docs.docker.com/compose/how-tos/gpu-support/) for the API. The host needs a compatible NVIDIA driver; CUDA libraries are included in the PyTorch image. First build downloads a large CUDA image. `docker compose config --quiet` validates configuration without an engine.

### Without Docker

From this directory in PowerShell (Python 3.12 or newer with a compatible PyTorch wheel):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install --force-reinstall --no-deps torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
$env:DQN_DEVICE = "cpu"
.\.venv\Scripts\python -m uvicorn connect4.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal:

```powershell
.\.venv\Scripts\python -m uvicorn connect4.simulation_service:app --host 127.0.0.1 --port 8080 --workers 1
```

Open http://localhost:8080 for the live viewer. Alternatively, run `python -m connect4.simulator` for the original terminal-only client. Run one simulator entry point at a time unless you deliberately want multiple training games. The viewer service reads `API_URL`, `MODEL_ONE`, `MODEL_TWO`, `SIM_SWAP_SIDES`, `SIM_INTERVAL`, and `SIM_RESULT_DELAY` from environment variables; Docker sets these automatically. The terminal client accepts `--swap-sides` (default) and `--no-swap-sides` to control starting positions.

The simulator alternates moves with no artificial delay by default; HTTP requests, rule evaluation, and training determine the actual speed. By default, `shared` learns against `rules`, and starting colors alternate between games. Rule moves are observed as opponent replies, just like human moves; only the AI's own actions enter replay, and final wins/losses are confirmed through `/finish`. For optional two-model self-play, use `--model-one alpha --model-two beta` (or `shared` for both). Use `--games 1000` for a fixed training run; `--games 0` runs continuously.

The CUDA wheel command above is for a compatible NVIDIA GPU/driver on Windows. See the [official PyTorch installer](https://pytorch.org/get-started/locally/) for other platforms. `DQN_DEVICE=cuda` requires CUDA and fails clearly if unavailable. The default `auto` uses CUDA when available and otherwise CPU; `cpu` forces CPU, and `cuda:0` selects a GPU explicitly. Both networks and all training batches run on the selected device; the simulator just sends HTTP requests. `GET /health`, `GET /models`, and move responses report the device. A separate CUDA toolkit installation is not needed for the prebuilt wheel. This tiny network may not run faster on a GPU because transfer and launch overhead can dominate.

Open http://127.0.0.1:8000/docs for interactive requests. Use **one API worker**, since weights, replay, and active games live in that process. Simulation and camera clients call the same server. For another device on a trusted LAN, bind to `0.0.0.0` and use the server's LAN address. This starter has no authentication; keep it on your trusted network.

## Camera / robot contract

The array is **`board[column][row]`**, with **row 0 at the top** and columns counted from the left. `0` means empty. Assign `1` to the player who starts the game and `-1` to the second player; the robot may be either. Labels stay fixed for the entire game. If your camera produces five rows of seven values, transpose that array first.

Send a unique `game_id` for each physical game. Keep it unchanged through the game, and use a different ID from the simulator. Request a move only on the robot's turn:

```json
POST /move
{
  "game_id": "physical-001",
  "board": [
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
  ],
  "player": 1,
  "model_id": "shared",
  "train": true,
  "explore": false
}
```

The response includes `column`, `predicted_board`, `done`, `winner`, `epsilon`, `loss`, `replay_size`, and `updates`. `winner` is `null` while ongoing, `0` for a draw, or the winning player's sign. `predicted_board` and `done` describe the result **if the suggested disk is placed successfully**. `loss` is the training loss, not whether the game was lost; it is `null` during replay warmup or with training disabled.

1. Read a stable camera board and call `/move` when it is the robot's turn.
2. Place exactly one disk in the returned column. Confirm it with the camera. Repeated identical requests return the cached action, but your robot controller must also remember which board/action it already executed so it never places a second disk on a retry.
3. Wait for the opponent's move, then send the next board using the same game ID. The API accepts its expected result plus at most one external opponent move. Two API-controlled players can instead alternate requests immediately after each placement.
4. **After either player wins or the board fills, send the actual final board to `/finish`.** Do this even when `/move` returned `done: true`, and even when the human won. This confirms the result and teaches both participating training players their terminal reward.

```json
POST /finish
{"game_id": "physical-001", "board": [[...], [...], [...], [...], [...], [...], [...]]}
```

The ellipses above stand for the seven actual columns; they are not valid JSON values. No `player` or `model_id` is needed for `/finish`; the session remembers them.

5. Delete the finished session with `DELETE /games/physical-001`. For a camera error, failed placement, or interrupted game, delete the session and start with a new ID and corrected board. Deleting discards outstanding moves without assigning a made-up loss. Already learned transitions remain in replay.

An existing valid midgame board may start a new session. Earlier moves cannot be learned from that snapshot. Keep each player's `model_id` and `train` setting fixed within a session. Identical `/move` and `/finish` retries are idempotent while the session exists. Don't reuse IDs or retry old requests after deleting their session. Board errors return 422; turn/order/session conflicts return 409. Floating disks, full columns, and inconsistent counts are rejected.

## Learning behavior

- The model takes a **7×5 matrix**, preserving `board[column][row]` throughout encoding and replay. Inputs are multiplied by the acting player's sign, so its own disks always look like `1`. Internally, three planes distinguish own discs, opponent discs, and legal landing cells.
- Three padded 3×3 convolution layers use 16 channels each, followed by a shared 1×1 scoring layer. Each column reads the score at its landing cell and adds a learned column bias. This gives **5,112 parameters**, a seven-cell receptive field, and seven Q-values without a flattened dense board input. The shared filters reuse patterns across positions. Greedy inference uses only the CNN and the legal-action mask; it has no hardcoded win/block override.
- Experience replay holds 20,000 transitions. Double DQN uses a target network updated every 100 optimizer steps, discount 0.99, Adam at 0.0003, Huber loss, and gradient clipping. Full columns are excluded from both action selection and bootstrap targets.
- Online training randomly reflects boards horizontally, including actions, next states, and legal-column masks. A training-only margin loss teaches immediate wins and single-threat blocks. Its weight is 0.5, and it stops pushing once a preferred move exceeds alternatives by 0.2. These labels never override the model's move during play.
- A transition runs from a player's turn to **that same player's next turn**, including the opponent's reply. Ongoing and draw rewards are 0. A confirmed win earns `1 - 0.01 * own_discs`; a confirmed loss earns `-1 + 0.01 * own_discs`. Faster wins and longer losses are preferred, with every win positive and every loss negative (at most 18 own discs). Rewards apply to simulation, human, and camera games. Older replay rewards are recomputed once from their normalized next boards on load; weights and experience are preserved. Both inference and replay always encode the acting AI as +1 and its opponent as -1, regardless of actual color.
- Each training `/move` performs one minibatch update when at least 32 transitions are available. The current action cannot supply a completed transition until its outcome or the next own turn is observed. `/finish` adds the final transitions and updates each affected model once. An empty board alone contains no reward to learn from.
- Simulation uses epsilon-greedy exploration, decaying by 1/50,000 per exploratory decision toward 0.05 (80% remains after the first 10,000 decisions from 100%). Camera moves default to `explore: false`, so the robot chooses the best legal model action while still learning from feedback. `train: false` disables learning for that participant, useful for evaluation.
- Model inference, updates, and checkpoints share a lock to prevent concurrent weight changes. Updates happen synchronously on the selected device, so there is no training queue or third service. Camera requests may briefly wait behind a simulation update or checkpoint.

Models are saved under `models/<model_id>.pt`, **after every training move and final-board report**, including replay warmup, and on graceful API shutdown. Saves use unique temporary files and atomically replace the previous checkpoint, retrying transient permission failures five times. A failed save is reported in diagnostics and retried on subsequent saves; it does not invalidate a game action. Until a save succeeds, recent learning exists only in memory. `POST /checkpoint` forces a save; `GET /models` shows loaded model statistics. Set `MODEL_DIR` to change the checkpoint directory. Checkpoints include both networks, optimizer state, replay, and exploration counters, and can be restored on CPU or CUDA. Active games and retry caches are **not** persisted: after an API restart, use new game IDs. A hard crash can lose an in-flight request. At most 1,000 game sessions and 32 loaded model IDs are allowed; delete old games, and restart the server to unload models.

You can keep the API running all day: a model ID always refers to the same in-memory agent and checkpoint file. Every game using `shared` contributes to that agent, even while a real game is active. The simulator runs until interrupted and deletes its completed sessions. Replay stays bounded at 20,000 transitions. Saving the whole replay on every move adds disk work and can reduce throughput during fast training.

### A small model within a 10,000-example budget

The supplied warm start uses 10,000 unique, rule-labeled legal positions: 3,000 immediate wins, 3,000 single-threat blocks, and 4,000 general rule-player choices. Training reuses these positions over multiple epochs, with mirrored examples; this is supervised teaching before continued Double DQN training, rather than a claim of mastery from 10,000 unassisted RL interactions. Validation and final tactical test positions, including their mirrors, are excluded from the training set.

The completed CPU run used the 5,112-parameter model and achieved **92.1875% immediate-win accuracy and 92.1875% blocking accuracy**, each on 256 held-out positions. It won **7 of 40** fixed-opponent benchmark games. These are basic tactical gains, not mastery of Connect Four. The run's detailed measurements are in `runtime/cnn-efficient-10000.json`. That warm-start run is historical. The subsequently requested reset has been consumed. No warm-start checkpoint is staged; current training uses the shared model created by that reset.

After `docker compose up --build -d`, the shared AI resumes its compatible checkpoint with the revised training algorithm. No automatic reset is performed by this update. The exploration preset buttons remain available. Old dense or earlier incompatible CNN checkpoints are discarded on load, as authorized for this architecture change. Freshly named models begin with random tiny-CNN weights.

To reproduce a fresh teaching run without touching the active server's model:

```powershell
.\.venv\Scripts\python -m connect4.train --model runtime/my-cnn.pt --observations 10000 --epochs 100 --device cpu
```

Use a new output path for each run. The script saves a checkpoint and a JSON report. It selects weights using validation tactics, then measures the final result on a separate 512-position test set. CPU is recommended here because GPU overhead outweighed the small model's computation in local checks; explicit CUDA execution remains supported.

## What else matters?

The main missing piece in a board-to-column request is **outcome feedback**: always report the final board, particularly when the opponent wins. A camera must also wait for stable detections and disk placement confirmation, and the robot must avoid executing an action twice. Camera vision and motor control are outside this project; their integration point is the HTTP contract above.

Self-play provides experience, but a fresh DQN starts with random weights and is not immediately a strong player. Moves run without delay by default; use `--interval 0.1` to slow down visible background play. Track performance against a fixed opponent rather than using self-play win counts as a measure of strength. Shared weights mean either side's experience benefits the deployed model immediately, and ongoing training can also temporarily reduce playing strength.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

Tests cover board rules, illegal boards and actions, retry handling, transitions between a player's own turns, win/loss/draw rewards, external human moves, checkpoint recovery, legal-action masking, and HTTP self-play with shared and separate model IDs.

The learning structure follows the [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), with game-specific two-turn transitions and Double DQN action selection. Synchronous FastAPI handlers use its [documented thread-pool behavior](https://fastapi.tiangolo.com/async/).

Enable **Three-player round robin** in Simulation configuration to select three player entries (at most one rule player). Use `shared`, `rules`, `shared` for two AI entries backed by the same model. Each pair plays 15 completed games. All three pairs play once per shuffled round, with no immediate repeated pair at round boundaries. The side-switch toggle still applies; with switching off the earlier roster entry starts. Human play pauses the rotation, interrupted games do not count, and applying a configuration restarts the rotation while retaining learned models. The benchmark remains fixed against rules regardless of the training roster; the new 50-game series is separate from older 40-game measurements.

The matchup matrix groups simulation results by underlying model: rows start, columns play second, and cells contain first-player wins / draws / second-player wins. With two shared AI entries and rules this is a 2x2 matrix; self-play counts in the shared/shared cell. Counts reset when applying configuration or restarting. Human games remain in the existing color totals only.

### Stability and progress diagnostics

New API experiences use up to three own-turn transitions: rewards are summed with discount 0.99 and the bootstrap uses `0.99 ** steps`. Queues are isolated per game and color, even for shared weights. Terminal results flush shortened tails; deleting an interrupted game discards its uncompleted tails. Existing one-step replay remains usable. The replay observation count tracks emitted experiences, so it can lag settled moves until tails flush.

The training rule-player curriculum begins at 50% rule moves / 50% random legal moves. Its rule probability is `0.5 + 0.5 * min(mean_last_five_benchmark_scores / 0.6, 1)`. Disable it in Simulation configuration for full-strength training opponents. Benchmarks always use full-strength rules, 25 games first and 25 second, without training or exploration. Charts also report a fixed 128-position immediate-win test, a 128-position block test, and average discounted completed-game reward.

Diagnostics expose TD loss, tactical-margin loss, sampled Q and target ranges, rejected updates, checkpoint failures, training/evaluation/request error counts, and the last service error. Predicted Q values beyond [-1.25, 1.25] raise a warning; they are not silently clipped. Non-finite losses and gradients cannot update weights. Training/evaluation failures are logged with tracebacks and counted; game transitions and retry caches remain usable. Counters are per service lifetime.

Reproduce the isolated baseline/revised comparison (three fresh seeds, identical 10,000-observation budgets):

```powershell
.\.venv\Scripts\python -m connect4.experiment --output runtime/new-comparison --observations 10000 --seeds 17 29 43
```

The baseline uses the previous unbounded tactical classification loss, 0.001 learning rate, one-step returns, fast exploration decay, and full rules. The revised profile uses the new settings and curriculum. Both use shared-weight round robin, side switching, and the same fixed benchmark. Results and separate checkpoints are written under the requested new output directory, never into live models. This comparison measures the combined changes; it does not isolate the contribution of each change.
