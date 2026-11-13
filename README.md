# dvdw-connect4

Reads a physical **Connect Four** board from a camera (Intel RealSense, any webcam, a network stream) or from a
recorded video, and serves the current position as a tiny JSON API.

Discs are found with **SAM 3** (Meta's text-prompted "Segment Anything" model) using the prompts `"red disc"` and
`"blue disc"`. The camera image is straightened with a one-time 4-corner calibration, every grid cell is classified
from the masks, and a temporal filter turns the noisy per-frame readings into a stable board.

```
camera / video ──► homography (4 clicked corners) ──► SAM 3 masks ──► per-cell classification
                                                                           │
            GET /state  ◄──  validity check  ◄──  temporal filter  ◄───────┘
```

## Contents

1. [Requirements](#1-requirements)
2. [Install on Linux](#2-install-on-linux)
3. [Install on Windows](#3-install-on-windows)
4. [Run](#4-run)
5. [Calibration](#5-calibration)
6. [The preview window](#6-the-preview-window)
7. [The REST API](#7-the-rest-api)
8. [How the board is stabilised](#8-how-the-board-is-stabilised)
9. [Command line options](#9-command-line-options)
10. [Troubleshooting](#10-troubleshooting)

## 1. Requirements

| | |
|---|---|
| **GPU** | An **NVIDIA GPU with a working driver** (`nvidia-smi` must work). SAM 3 runs on CUDA only; there is no CPU fallback. CUDA 12.6 or newer. |
| **Python** | **3.12 or newer** (a conda environment is recommended), as required by SAM 3. |
| **PyTorch** | 2.7 or newer with CUDA support. |
| **Model access** | The SAM 3 checkpoints are **gated** on Hugging Face. Request access to [`facebook/sam3`](https://huggingface.co/facebook/sam3) with your Hugging Face account and wait for approval *before* the first run. |
| **Camera** (optional) | Intel RealSense (D4xx), any webcam, or an RTSP/HTTP stream. You can also just use a video file. |
| **Disk** | A few GB for PyTorch and the SAM 3 checkpoint. |

Where it was developed and tested: Linux (Ubuntu), Python 3.12, PyTorch 2.7.0 + CUDA 12.6, SAM 3 installed from a
clone of its repository with `pip install -e .`, NVIDIA RTX 3500 Ada laptop GPU. On that machine one detection pass
(both colours) takes about **0.25 s**, so the board updates about four times per second. The camera preview itself
runs at full frame rate. The Windows instructions follow the same steps but were not run on Windows by the
maintainer.

## 2. Install on Linux

The recommended way is to install SAM 3 exactly as described in its own repository
(<https://github.com/facebookresearch/sam3#installation>) and then add this project's few extra dependencies.

```bash
# 1. get this project
git clone https://github.com/JVanherck/dvdw-connect4.git
cd dvdw-connect4

# 2. create an environment (SAM 3 needs Python >= 3.12)
conda create -n 4x4 python=3.12 -y
conda activate 4x4

# 3. PyTorch with CUDA (as in the SAM 3 instructions; 2.7+ with CUDA 12.6+ works, 2.7.0+cu126 was tested)
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. SAM 3: clone its repository and install it in editable mode
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

# 5. the rest (OpenCV, Flask, RealSense, and imports SAM 3 does not declare)
pip install -r requirements.txt

# 6. log in to Hugging Face (needs approved access to facebook/sam3, see above)
hf auth login            # older huggingface_hub versions: huggingface-cli login

# 7. sanity check: must print True
python -c "import torch; print(torch.cuda.is_available())"
```

Notes:

* The `sam3/` clone inside the project folder is fine (it is in `.gitignore`, and `connect4_reader.py` makes sure
  it does not shadow the installed package). Cloning it somewhere else works just as well.
* **RealSense on Linux:** `pyrealsense2` comes from pip, but to open the camera without `sudo` you need the udev
  rules. Either install `librealsense2-udev-rules` from Intel's apt repository
  ([instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)) or copy
  `config/99-realsense-libusb.rules` from the librealsense repository into `/etc/udev/rules.d/`, then run
  `sudo udevadm control --reload-rules && sudo udevadm trigger` and re-plug the camera. Use a **USB 3** port and cable.
* **Display:** the calibration and preview windows need a desktop session. Over SSH, calibrate once on the machine
  itself (or with X forwarding) and afterwards run with `--no-display`.

## 3. Install on Windows

Use **Anaconda/Miniconda Prompt** or PowerShell with conda available. Install the latest **NVIDIA driver** first;
`nvidia-smi` must work in a terminal. You do *not* need to install the CUDA toolkit separately, the PyTorch wheels
bundle the CUDA runtime. You also need **Git for Windows**.

```powershell
# 1. get this project
git clone https://github.com/JVanherck/dvdw-connect4.git
cd dvdw-connect4

# 2. create an environment (SAM 3 needs Python >= 3.12)
conda create -n 4x4 python=3.12 -y
conda activate 4x4

# 3. PyTorch with CUDA (as in the SAM 3 instructions)
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. SAM 3: clone its repository and install it in editable mode
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

# 5. the rest, plus triton (SAM 3 imports it; there is no official Windows wheel, so use triton-windows)
pip install -r requirements.txt
pip install triton-windows

# 6. log in to Hugging Face (needs approved access to facebook/sam3, see above)
hf auth login

# 7. sanity check: must print True
python -c "import torch; print(torch.cuda.is_available())"
```

Notes:

* If `torch.cuda.is_available()` prints `False`, you most likely got a CPU-only PyTorch build, see
  [Troubleshooting](#10-troubleshooting).
* **RealSense on Windows:** `pip install pyrealsense2` (already in `requirements.txt`) is enough to open the
  camera. Installing the [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases) is optional
  but useful for the Viewer to check that the camera works. Use a USB 3 port.
* To read the API from PowerShell use `Invoke-RestMethod http://localhost:8000/state` (or `curl.exe`, not `curl`).
* If another machine should reach the API, allow Python through the Windows firewall for port 8000.

## 4. Run

Activate the environment first (`conda activate 4x4`). Commands are the same on Linux and Windows.
The default board is the standard **7 columns × 6 rows**; use `--rows` / `--cols` for other sizes.

```bash
# a recorded video (loops when it ends, plays in real time)
python connect4_reader.py --video my_game.mp4

# a 7x5 board
python connect4_reader.py --video my_game.mp4 --rows 5

# a live Intel RealSense camera (default when no source is given)
python connect4_reader.py

# a live webcam (index 0, 1, ...) or a network stream
python connect4_reader.py --camera 0
python connect4_reader.py --camera rtsp://192.168.1.50:554/stream

# no windows at all (needs an existing calibration file)
python connect4_reader.py --video my_game.mp4 --no-display
```

The first start takes a while because SAM 3 and its checkpoint are loaded (the API answers immediately with
`valid: false` until the model is ready). Then query it:

```bash
curl http://localhost:8000/state
```

Press `q` or `Esc` in the preview window to quit (or `Ctrl+C` in the terminal when running with `--no-display`).

## 5. Calibration

The script needs to know where the board is in the image. On the **first run** (no `calibration.json` yet) a
window opens with the live image:

1. Click the **4 outer corners of the grid** in this order: **top-left, top-right, bottom-right, bottom-left**
   (the on-screen prompt tells you which one is next).
2. A second window shows the straightened board with the grid lines drawn on it; every cell should sit nicely
   on a hole.
3. Press `Enter` or `Space` to accept, or `R` to start over. `q`/`Esc` aborts.

The corners are saved to `calibration.json` next to the script and reused on the next start.
Pass `--recalibrate` to click new corners, or `--calibration other.json` to keep several setups.
**Calibrate again whenever the camera or the board is moved.** Calibration is specific to your camera position,
which is why `calibration.json` is not part of the repository.

Tips: get the whole board in view, keep it reasonably well lit, and avoid strong reflections on the discs.

## 6. The preview window

The window shows the straightened board with the detected, time-filtered discs circled in their colour.
By default the raw **SAM 3 masks** are overlaid (red prompt → yellow, blue prompt → cyan, with outlines), which is
useful to see what the model actually sees. Because the circles show the *filtered* board and the masks the *raw*
detections, you can watch the filter ignore hands, falling discs and dropouts.

| Key | Action |
|---|---|
| `m` | toggle the mask overlay (magenta outlines = detected by SAM but rejected by the depth filter) |
| `d` | toggle the per-cell depth numbers (fraction of the hole that depth says is filled) |
| `q` / `Esc` | quit |

The text in the top-left corner shows `valid` or the reason the current board is physically impossible.
Start without the overlay with `--hide-masks`.

## 7. The REST API

`GET http://<host>:8000/state` returns:

```json
{
  "valid": true,
  "timestamp": "2026-11-10T11:51:30.396951+00:00",
  "game_id": 1,
  "board": [
    ["X", "X", "X", "X", "X", "X", "X"],
    ["X", "X", "X", "X", "X", "X", "X"],
    ["X", "X", "X", "X", "X", "X", "X"],
    ["X", "X", "B", "X", "X", "X", "X"],
    ["X", "R", "R", "B", "R", "X", "X"]
  ]
}
```

| Field | Meaning |
|---|---|
| `board` | 2D array, **row 0 is the top row**. `X` = empty, `R` = red, `B` = blue. This is the time-filtered board. |
| `valid` | `true` if the board is a physically reachable position: no floating discs, red/blue counts differ by at most one, and not both players have four in a row. Check this before trusting `board`. |
| `timestamp` | UTC time (ISO 8601) of the last processed frame. `null` until the first reading. |
| `game_id` | Starts at 1 and increases every time a new game is detected (see below). |

While the model is still loading you get `valid: false`, `timestamp: null` and an empty board.
The server listens on `0.0.0.0:8000` by default (`--host`, `--port`).

## 8. How the board is stabilised

Raw SAM 3 output flickers: a hand or leg in front of the board, a disc that is half visible while falling, or a
random model dropout can make a disc vanish for a few frames. The tracker (`BoardTracker`) filters this:

* **Gravity.** A disc only counts if it is on the bottom row or the cell below it is already filled. A disc
  sliding down a column is ignored until it lands.
* **Confirmation.** A new disc must be seen continuously for 0.5 s (`--confirm-seconds`) before it is added.
* **No removals.** A settled disc is never removed by a hand, a leg or a detection dropout.
* **Colour flips.** A settled disc only changes colour after 5 s of continuous disagreement (`--flip-seconds`).
* **New game.** If **no disc at all** is detected for 5 consecutive seconds (`--reset-seconds`), the game is
  considered restarted: the board is cleared and `game_id` goes up by one.

## 8a. Depth filter (RealSense only)

With a RealSense the depth stream is aligned to the colour image and used in two ways:

* **Only things in the board are discs.** At startup the board's front face is fitted as a plane from depth (the grid
  area minus the holes). Every detected pixel that is further than `--depth-behind` behind that plane (a red/blue
  poster, chair or wall seen through an empty hole) or further than `--depth-front` in front of it (a hand, arm or leg)
  is discarded before the cells are classified.
* **Consistent colours.** A SAM detection only counts if the cell really has that colour (red/blue hue check), and
  where depth says a hole is physically filled, the hue alone counts as evidence. A frame where SAM misses a disc
  therefore no longer loses it, and a dark hole that SAM mistakes for a disc is rejected.

The startup message prints the fitted plane distance and fit error. If the plane cannot be fitted, or you use a
video/webcam/stream, the depth filter is switched off and only SAM is used. The defaults (2.5 cm behind, 3 cm in front) come from a live test where the discs sat 0–1 cm behind the front
face while a person's jeans behind the board were 3–10 cm behind it. If real discs get rejected, increase
`--depth-behind`; the `d` key shows how filled each cell is.

## 9. Command line options

| Option | Default | Description |
|---|---|---|
| `--video FILE` | | Read a video file instead of a live camera (plays in real time, loops at the end). |
| `--camera SRC` | | Live webcam index (`0`) or stream URL (`rtsp://…`, `http://…`) instead of the RealSense. |
| `--rows N` | `6` | Rows of the board. |
| `--cols N` | `7` | Columns of the board. |
| `--width`, `--height`, `--fps` | `1280`, `720`, `30` | Requested camera resolution and frame rate (RealSense and webcam). |
| `--calibration FILE` | `calibration.json` next to the script | Where the board corners are stored. |
| `--recalibrate` | off | Ignore the saved corners and click new ones. |
| `--red-prompt` | `red disc` | SAM 3 text prompt for red discs. |
| `--blue-prompt` | `blue disc, blue circle` | SAM 3 text prompt(s) for blue discs; a comma-separated list is combined (union). |
| `--confidence` | `0.4` | SAM 3 detection threshold. |
| `--min-coverage` | `0.5` | Fraction of a cell's centre a mask must cover to count. |
| `--confirm-seconds` | `0.5` | How long a new disc must be seen before it is added. |
| `--flip-seconds` | `5.0` | How long a different colour must persist before a settled disc flips. |
| `--reset-seconds` | `5.0` | No discs detected for this long means a new game: clear the board. |
| `--no-depth` | off | Do not use the RealSense depth stream (depth filter off). |
| `--depth-behind` | `0.025` | Metres behind the board's front face that still count as a disc; anything further away (poster, wall) is rejected. |
| `--depth-front` | `0.03` | Metres in front of the board's front face that still count as a disc; anything closer (hand, arm) is rejected. |
| `--hide-masks` | off | Start with the mask overlay off (toggle with `m`). |
| `--no-display` | off | No preview window. Needs an existing calibration file. |
| `--host`, `--port` | `0.0.0.0`, `8000` | Address of the REST API. |

Prompt tuning notes: on the test video the short prompts `red disc` / `blue disc` worked best. Descriptive
variants ("dark blue disc", "teal disc", "blue game piece") found far fewer discs, and lowering `--confidence`
to 0.08 produced many false detections (0.15 and 0.25 gave no improvement).

## 10. Troubleshooting

**`Torch not compiled with CUDA enabled` / `torch.cuda.is_available()` is `False`**
You have a CPU-only PyTorch. Remove it and reinstall the CUDA build:
`pip uninstall -y torch torchvision`, then repeat the PyTorch step of the installation. Also check that `nvidia-smi` works.

**`SAM3 needs CUDA, but torch … cannot see a GPU`**
Same cause as above, or the NVIDIA driver is not loaded.

**`401` / `403` / "gated repo" error while loading the model**
You have not been granted access to `facebook/sam3`, or you are not logged in. Request access on Hugging Face,
then run `hf auth login`.

**`ModuleNotFoundError: No module named 'pkg_resources'`**
`setuptools` 82+ removed it. `pip install "setuptools<81"` (already pinned in `requirements.txt`).

**NumPy errors / `numpy.core.multiarray failed to import`**
SAM 3 needs `numpy<2` and OpenCV 4.10/4.11 matches it: `pip install "numpy<2" "opencv-python>=4.10,<4.12"`.

**`TypeError: expected str, bytes or os.PathLike object, not NoneType` inside `sam3/model_builder.py`**
A folder called `sam3/` (for example a git clone of SAM 3) sits in the directory you run from and shadows the
installed package. `connect4_reader.py` already guards against this; if you see it from your own script, run it
from another directory or delete/rename the clone.

**`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`**
Inference must run under bf16 autocast. `connect4_reader.py` does this; it only appears if you call SAM 3
yourself without `torch.autocast("cuda", dtype=torch.bfloat16)`.

**`cannot start the RealSense camera (No device connected)`**
The camera is not detected. Check the USB 3 cable and port, that no other program (e.g. the RealSense Viewer)
is using it, and on Linux that the udev rules are installed. Use `--camera 0` for a webcam or `--video FILE`.

**`no frame from the RealSense for 2 s, still waiting...`**
The stream stalled (loose cable, USB power). It keeps retrying and continues once frames return.

**`--no-display needs an existing calibration file`**
Run once with a display to calibrate, or copy a `calibration.json` made on another machine with the same
camera position.

**The API shows discs that are not there / misses discs**
Check the mask overlay (`m`) to see what SAM 3 detects. Re-run `--recalibrate` if the grid is not aligned with
the holes. Reduce reflections and make sure the whole board is lit evenly. `--min-coverage` and `--confidence`
can be tuned, but the temporal filter already removes most one-off errors.

**`Address already in use`**
Another instance is still running. Stop it or use `--port 8001`.

**Preview window does not open on a remote Linux machine**
OpenCV needs a display. Calibrate locally, then run with `--no-display` and read the API.

## Files

| File | Purpose |
|---|---|
| `connect4_reader.py` | The application: camera input, calibration, SAM 3 detection, time filter, REST API, preview. |
| `requirements.txt` | App dependencies (OpenCV, Flask, RealSense, …) to install after PyTorch and SAM 3. |
