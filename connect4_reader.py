#!/usr/bin/env python3
"""Read a Connect Four board from a live camera (RealSense, webcam or stream) or a video and serve it as JSON.

1. Click the four board corners once (saved to calibration.json) -> homography.
2. SAM3 open-vocabulary prompts ("red disc", "blue disc") find the discs in the
   rectified board image; each grid cell is classified by mask coverage.
3. The grid is checked for physical plausibility (gravity, disc counts, winners).
4. GET /state returns {valid, timestamp, game_id, board}; board is a 2D array of X (empty), R, B.
"""
import sys
from pathlib import Path as _Path

# The cloned ./sam3 repo folder would shadow the installed sam3 package as an
# empty namespace package, so keep this script's directory off sys.path.
_here = _Path(__file__).resolve().parent
sys.path = [p for p in sys.path if _Path(p or ".").resolve() != _here]

import argparse
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify

COLS, ROWS = 7, 6
CELL = 100  # pixels per cell in the rectified image
WARP_SIZE = (COLS * CELL, ROWS * CELL)
COLORS = ("red", "blue")
DRAW_BGR = {"red": (0, 0, 255), "blue": (255, 120, 0)}


# --------------------------------------------------------------------------- camera
class RealSenseCamera:
    def __init__(self, width, height, fps):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            raise SystemExit(f"cannot start the RealSense camera ({e}). Is it plugged in (USB 3) and not in use "
                             "by another program? Use --camera N for a webcam or --video FILE for a recording.")
        self.warned = False

    def read(self):
        ok, frames = self.pipeline.try_wait_for_frames(2000)  # a dropped frame must not kill the app
        if not ok:
            if not self.warned:
                print("warning: no frame from the RealSense for 2 s, still waiting...")
                self.warned = True
            return None
        self.warned = False
        frame = frames.get_color_frame()
        return np.asanyarray(frame.get_data()).copy() if frame else None

    def stop(self):
        self.pipeline.stop()


class StreamCamera:
    """Any live OpenCV source: a webcam index ("0") or a stream URL (rtsp://, http://...)."""

    def __init__(self, source, width, height, fps):
        is_index = str(source).isdigit()
        self.cap = cv2.VideoCapture(int(source) if is_index else str(source))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera/stream {source!r}")
        if is_index:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(0.05)
            return None
        return frame

    def stop(self):
        self.cap.release()


class VideoCamera:
    """Video file with the same interface as RealSenseCamera; loops at the end."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.delay = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.last = 0.0

    def read(self):
        wait = self.last + self.delay - time.time()  # play in real time
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self):
        self.cap.release()


# --------------------------------------------------------------------------- calibration
def draw_grid(img):
    for c in range(COLS + 1):
        cv2.line(img, (c * CELL, 0), (c * CELL, WARP_SIZE[1]), (0, 255, 0), 1)
    for r in range(ROWS + 1):
        cv2.line(img, (0, r * CELL), (WARP_SIZE[0], r * CELL), (0, 255, 0), 1)


def homography(corners):
    dst = np.float32([[0, 0], [WARP_SIZE[0], 0], [WARP_SIZE[0], WARP_SIZE[1]], [0, WARP_SIZE[1]]])
    return cv2.getPerspectiveTransform(np.float32(corners), dst)


def select_corners(camera):
    """Let the user click the 4 outer corners of the 7x6 cell grid on the live view."""
    win = "Calibration"
    pts = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    while True:
        frame = camera.read()
        if frame is None:
            cv2.waitKey(1)
            continue
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 6, (0, 255, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.int32(pts)], len(pts) == 4, (0, 255, 255), 2)
        if len(pts) < 4:
            msg = f"Click {names[len(pts)]} corner of the grid (slots at the top)"
        else:
            msg = "ENTER/SPACE = accept, R = reset"
            warped = cv2.warpPerspective(frame, homography(pts), WARP_SIZE)
            draw_grid(warped)
            cv2.imshow("Rectified preview", warped)
        cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            pts.clear()
            cv2.destroyWindow("Rectified preview")
        elif key in (13, 32) and len(pts) == 4:
            break
        elif key in (27, ord("q")):
            raise SystemExit("Calibration aborted")
    cv2.destroyWindow(win)
    cv2.destroyWindow("Rectified preview")
    return pts


def load_or_calibrate(camera, path, force):
    if path.exists() and not force:
        return json.loads(path.read_text())["corners"]
    corners = [list(map(float, p)) for p in select_corners(camera)]
    path.write_text(json.dumps({"corners": corners}, indent=2))
    return corners


# --------------------------------------------------------------------------- SAM3
class Sam3Segmenter:
    """Open-vocabulary text-prompted segmentation; returns one boolean mask per prompt."""

    def __init__(self, confidence):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"SAM3 needs CUDA, but torch {torch.__version__} cannot see a GPU "
                             "(CPU-only build or NVIDIA driver not loaded; check `nvidia-smi`)")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=confidence)

    def segment(self, image_bgr, prompts):
        import torch
        from PIL import Image

        h, w = image_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        out = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(pil)
            for name, prompt in prompts.items():
                self.processor.reset_all_prompts(state)
                masks = self.processor.set_text_prompt(state=state, prompt=prompt)["masks"]
                if masks is None or len(masks) == 0:
                    out[name] = np.zeros((h, w), bool)
                else:
                    out[name] = masks.reshape(-1, h, w).cpu().numpy().astype(bool).any(axis=0)
        return out


# --------------------------------------------------------------------------- grid logic
_yy, _xx = np.mgrid[:CELL, :CELL]
_DISC = (_xx - CELL / 2) ** 2 + (_yy - CELL / 2) ** 2 <= (0.32 * CELL) ** 2


def classify_cells(masks, min_coverage):
    """grid[row][col] in {'empty','red','blue'}; row 0 is the top row."""
    grid = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            sl = (slice(r * CELL, (r + 1) * CELL), slice(c * CELL, (c + 1) * CELL))
            cov = {k: float(m[sl][_DISC].mean()) for k, m in masks.items()}
            best = max(cov, key=cov.get)
            row.append(best if cov[best] >= min_coverage else "empty")
        grid.append(row)
    return grid


def find_winners(grid):
    winners = set()
    for r in range(ROWS):
        for c in range(COLS):
            color = grid[r][c]
            if color == "empty":
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + 3 * dr, c + 3 * dc
                if 0 <= er < ROWS and 0 <= ec < COLS and all(
                    grid[r + i * dr][c + i * dc] == color for i in range(4)
                ):
                    winners.add(color)
    return winners


def validate(grid):
    """Return (errors, winner). An empty error list means the position is reachable."""
    errors = []
    for c in range(COLS):
        gap_below = False
        for r in range(ROWS - 1, -1, -1):  # bottom -> top
            if grid[r][c] == "empty":
                gap_below = True
            elif gap_below:
                errors.append(f"floating disc in column {c + 1}, row {r + 1} (from top)")
                break
    n = {k: sum(row.count(k) for row in grid) for k in COLORS}
    if abs(n["red"] - n["blue"]) > 1:
        errors.append(f"disc counts impossible (red={n['red']}, blue={n['blue']})")
    winners = find_winners(grid)
    if len(winners) > 1:
        errors.append("both players have four in a row")
    winner = next(iter(winners)) if len(winners) == 1 else None
    if winner:
        other = "blue" if winner == "red" else "red"
        if n[winner] < n[other]:
            errors.append(f"{winner} has four in a row but {other} moved last")
    return errors, winner


# --------------------------------------------------------------------------- temporal filter
class BoardTracker:
    """Turns noisy per-frame readings into a stable board.

    * A disc only counts if it is supported (bottom row, or the cell below is occupied), so a
      disc sliding down a column is ignored until it lands.
    * A new disc must be seen continuously for `confirm_s` seconds before it is added.
    * Settled discs are never removed by hands/legs/SAM dropouts. Their colour only flips after
      `flip_s` seconds of continuous disagreement.
    * If no disc at all is detected for `reset_s` seconds, the game restarted: clear the board.
    """

    def __init__(self, confirm_s=0.5, flip_s=5.0, reset_s=5.0):
        self.confirm_s, self.flip_s, self.reset_s = confirm_s, flip_s, reset_s
        self.game_id = 1
        self._clear()

    def reset(self):
        """New game: clear the board and bump game_id."""
        self._clear()
        self.game_id += 1

    def _clear(self):
        self.board = [["empty"] * COLS for _ in range(ROWS)]
        self.pending = {}   # (r, c) -> (colour, first_seen)
        self.differs = {}   # (r, c) -> (colour_seen, first_seen) for settled cells that disagree
        self.none_since = None

    def update(self, raw, now):
        """raw: grid of 'empty'/'red'/'blue' for this frame. Returns the settled board."""
        if not any(v != "empty" for row in raw for v in row):
            self.none_since = self.none_since if self.none_since is not None else now
            if now - self.none_since >= self.reset_s and any(v != "empty" for row in self.board for v in row):
                self.reset()
                self.none_since = now
        else:
            self.none_since = None

        # supported cells of this frame (chain bottom -> top)
        sup = [[False] * COLS for _ in range(ROWS)]
        for c in range(COLS):
            for r in range(ROWS - 1, -1, -1):
                if raw[r][c] == "empty":
                    continue
                below = r == ROWS - 1 or self.board[r + 1][c] != "empty" or sup[r + 1][c]
                sup[r][c] = below

        # additions, processed bottom -> top so a whole column can settle in one tick
        for c in range(COLS):
            for r in range(ROWS - 1, -1, -1):
                if self.board[r][c] != "empty":
                    continue
                key = (r, c)
                colour = raw[r][c]
                if colour == "empty" or not sup[r][c]:
                    self.pending.pop(key, None)
                    continue
                if key not in self.pending or self.pending[key][0] != colour:
                    self.pending[key] = (colour, now)
                supported_by_board = r == ROWS - 1 or self.board[r + 1][c] != "empty"
                if supported_by_board and now - self.pending[key][1] >= self.confirm_s:
                    self.board[r][c] = colour
                    del self.pending[key]

        # colour flips of settled discs (never removals)
        for r in range(ROWS):
            for c in range(COLS):
                cur, seen = self.board[r][c], raw[r][c]
                key = (r, c)
                if cur == "empty" or seen in ("empty", cur):
                    self.differs.pop(key, None)
                    continue
                if key not in self.differs or self.differs[key][0] != seen:
                    self.differs[key] = (seen, now)
                if now - self.differs[key][1] >= self.flip_s:
                    self.board[r][c] = seen
                    del self.differs[key]
        return [row[:] for row in self.board]


# --------------------------------------------------------------------------- shared state
SYMBOL = {"empty": "X", "red": "R", "blue": "B"}


class BoardState:
    """Latest reading. The public API exposes only valid / timestamp / game_id / board."""

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "starting"   # for the preview window only
        self.errors = []           # for the preview window / console only
        self.observed = None       # filtered board in colour names, for the preview window
        self.api = {"valid": False, "timestamp": None, "game_id": 1,
                    "board": [["X"] * COLS for _ in range(ROWS)]}

    def update(self, observed, errors, game_id):
        with self.lock:
            self.status, self.errors, self.observed = "ok", errors, observed
            self.api = {
                "valid": not errors,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "game_id": game_id,
                "board": [[SYMBOL[v] for v in row] for row in observed],
            }

    def set_status(self, status):
        with self.lock:
            self.status = status

    def snapshot(self):
        with self.lock:
            return dict(self.api)

    def preview(self):
        with self.lock:
            return {"status": self.status, "observed": self.observed, "errors": list(self.errors)}


def make_app(state):
    app = Flask(__name__)

    @app.get("/state")
    def get_state():
        return jsonify(state.snapshot())

    return app


# --------------------------------------------------------------------------- detection worker
class Detector(threading.Thread):
    def __init__(self, segmenter_factory, state, prompts, min_coverage, tracker):
        super().__init__(daemon=True)
        self.factory, self.state, self.prompts = segmenter_factory, state, prompts
        self.min_coverage, self.tracker = min_coverage, tracker
        self.lock = threading.Lock()
        self.frame, self.frame_id = None, 0
        self.observed = None
        self.masks = None  # latest SAM masks {colour: bool array}, for the preview overlay
        self.stop_event = threading.Event()

    def submit(self, warped):
        with self.lock:
            self.frame, self.frame_id = warped, self.frame_id + 1

    def run(self):
        self.state.set_status("loading SAM3 model")
        segmenter = self.factory()
        self.state.set_status("waiting for frames")
        last_id = 0
        while not self.stop_event.is_set():
            with self.lock:
                frame, fid = self.frame, self.frame_id
            if frame is None or fid == last_id:
                time.sleep(0.01)
                continue
            last_id = fid
            masks = segmenter.segment(frame, self.prompts)
            self.masks = masks
            raw = classify_cells(masks, self.min_coverage)
            observed = self.tracker.update(raw, time.monotonic())
            errors, _ = validate(observed)
            self.observed = observed
            self.state.update(observed, errors, self.tracker.game_id)


# --------------------------------------------------------------------------- main
def annotate(warped, observed, masks=None):
    view = warped.copy()
    if masks:  # tint SAM masks and outline them (red prompt -> red, blue prompt -> cyan)
        for colour, bgr in (("red", (0, 255, 255)), ("blue", (255, 255, 0))):
            m = masks.get(colour)
            if m is None or m.shape != view.shape[:2]:
                continue
            view[m] = (0.5 * view[m] + 0.5 * np.array(bgr)).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, cnts, -1, bgr, 2)
    draw_grid(view)
    if observed:
        for r in range(ROWS):
            for c in range(COLS):
                if observed[r][c] != "empty":
                    cv2.circle(view, (c * CELL + CELL // 2, r * CELL + CELL // 2), int(CELL * 0.35),
                               DRAW_BGR[observed[r][c]], 3)
    return view


def main():
    global ROWS, COLS, WARP_SIZE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=ROWS, help="grid rows (standard board: 6)")
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--video", type=Path, help="read this video file instead of a live camera (loops at the end)")
    ap.add_argument("--camera", help="live webcam index (e.g. 0) or stream URL (rtsp://...) instead of the RealSense")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--calibration", type=Path, default=Path(__file__).with_name("calibration.json"))
    ap.add_argument("--recalibrate", action="store_true", help="ignore the saved corners and click new ones")
    ap.add_argument("--red-prompt", default="red disc")
    ap.add_argument("--blue-prompt", default="blue disc")
    ap.add_argument("--confidence", type=float, default=0.4, help="SAM3 detection threshold")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="fraction of a cell centre a mask must cover")
    ap.add_argument("--confirm-seconds", type=float, default=0.5, help="a new disc must be seen this long before it is added")
    ap.add_argument("--flip-seconds", type=float, default=5.0, help="a settled disc changes colour only after this long")
    ap.add_argument("--reset-seconds", type=float, default=5.0, help="no discs detected for this long = game restarted, clear the board")
    ap.add_argument("--hide-masks", action="store_true", help="start with the SAM mask overlay off (toggle with m)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-display", action="store_true", help="skip the preview window (needs saved calibration)")
    args = ap.parse_args()
    ROWS, COLS = args.rows, args.cols
    WARP_SIZE = (COLS * CELL, ROWS * CELL)

    if args.video:
        camera = VideoCamera(args.video)
    elif args.camera is not None:
        camera = StreamCamera(args.camera, args.width, args.height, args.fps)
    else:
        camera = RealSenseCamera(args.width, args.height, args.fps)
    try:
        if args.no_display and not args.calibration.exists():
            raise SystemExit("--no-display needs an existing calibration file")
        H = homography(load_or_calibrate(camera, args.calibration, args.recalibrate))

        state = BoardState()
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        app = make_app(state)
        threading.Thread(target=lambda: app.run(args.host, args.port, threaded=True), daemon=True).start()
        print(f"REST API: http://{args.host}:{args.port}/state")

        detector = Detector(lambda: Sam3Segmenter(args.confidence), state,
                            {"red": args.red_prompt, "blue": args.blue_prompt},
                            args.min_coverage,
                            BoardTracker(args.confirm_seconds, args.flip_seconds, args.reset_seconds))
        detector.start()

        show_masks = not args.hide_masks
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            warped = cv2.warpPerspective(frame, H, WARP_SIZE)
            detector.submit(warped)
            if args.no_display:
                continue
            snap = state.preview()
            view = annotate(warped, snap["observed"], detector.masks if show_masks else None)
            label = snap["status"] if snap["status"] != "ok" else ("; ".join(snap["errors"]) or "valid")
            cv2.putText(view, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Connect Four", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                show_masks = not show_masks
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
