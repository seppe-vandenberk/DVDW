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
import warnings
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
    """RealSense colour stream plus (when available) depth aligned to colour; `self.depth` is metres, 0 = invalid."""

    def __init__(self, width, height, fps, use_depth=True):
        import pyrealsense2 as rs

        self.rs, self.depth, self.align, self.depth_scale = rs, None, None, None
        self.pipeline = rs.pipeline()
        attempts = [(width, height, fps), (848, 480, 30), (640, 480, 30)] if use_depth else []
        started = False
        for dw, dh, dfps in attempts:
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
            try:
                profile = self.pipeline.start(config)
            except RuntimeError:
                continue
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            started = True
            break
        if not started:
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                self.pipeline.start(config)
            except RuntimeError as e:
                raise SystemExit(f"cannot start the RealSense camera ({e}). Is it plugged in (USB 3) and not in use "
                                 "by another program? Use --camera N for a webcam or --video FILE for a recording.")
            if use_depth:
                print("warning: could not start the depth stream, running without the depth filter")
        self.warned = False

    @property
    def has_depth(self):
        return self.align is not None

    def read(self):
        ok, frames = self.pipeline.try_wait_for_frames(2000)  # a dropped frame must not kill the app
        if not ok:
            if not self.warned:
                print("warning: no frame from the RealSense for 2 s, still waiting...")
                self.warned = True
            return None
        self.warned = False
        if self.align is not None:
            frames = self.align.process(frames)
            depth = frames.get_depth_frame()
            self.depth = (np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale) if depth else None
        frame = frames.get_color_frame()
        return np.asanyarray(frame.get_data()).copy() if frame else None

    def stop(self):
        self.pipeline.stop()


class StreamCamera:
    """Any live OpenCV source: a webcam index ("0") or a stream URL (rtsp://, http://...)."""

    depth, has_depth = None, False

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

    depth, has_depth = None, False

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
                union = np.zeros((h, w), bool)
                for p in (x.strip() for x in prompt.split(",") if x.strip()):  # "a, b" = union of both prompts
                    self.processor.reset_all_prompts(state)
                    masks = self.processor.set_text_prompt(state=state, prompt=p)["masks"]
                    if masks is not None and len(masks) > 0:
                        union |= masks.reshape(-1, h, w).cpu().numpy().astype(bool).any(axis=0)
                out[name] = union
        return out


# --------------------------------------------------------------------------- grid logic
_yy, _xx = np.mgrid[:CELL, :CELL]
_DISC = (_xx - CELL / 2) ** 2 + (_yy - CELL / 2) ** 2 <= (0.32 * CELL) ** 2
_HOLE = (_xx - CELL / 2) ** 2 + (_yy - CELL / 2) ** 2 <= (0.46 * CELL) ** 2  # whole hole, incl. its rim


class DepthGate:
    """Uses the depth camera to tell what is in the board's plane from what is in front of or behind it.

    The board's front face is a plane, so 1/z is linear in image coordinates; it is fitted on the face pixels
    (grid area minus the holes). `evaluate` then returns, in the rectified view, the depth of every pixel relative
    to that plane in metres: > 0 is behind the front face, < 0 is in front of it, NaN is unknown.
    """

    def __init__(self, H, image_wh):
        self.H, self.size, self.z_plane = H, image_wh, None
        self.median_z = None

    def fit(self, depth_frames):
        w, h = self.size
        stack = np.stack(depth_frames)
        stack = np.where(stack > 0, stack, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.nanmedian(stack, axis=0)
        holes = np.tile(_HOLE, (ROWS, COLS))
        face = cv2.erode((~holes).astype(np.uint8), np.ones((9, 9), np.uint8))
        face = cv2.warpPerspective(face, self.H, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        ys, xs = np.nonzero((face > 0) & np.isfinite(z) & (z > 0.2) & (z < 8.0))
        if len(xs) < 3000:
            return False
        if len(xs) > 40000:
            keep = np.random.default_rng(0).choice(len(xs), 40000, replace=False)
            ys, xs = ys[keep], xs[keep]
        A = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
        inv = 1.0 / z[ys, xs].astype(np.float64)
        keep = np.ones(len(xs), bool)
        for _ in range(5):  # least squares with outlier rejection (discs' rims, edges, hands)
            coef = np.linalg.lstsq(A[keep], inv[keep], rcond=None)[0]
            res = A @ coef - inv
            sigma = max(1.4826 * np.median(np.abs(res[keep])), 1e-5)
            keep = np.abs(res) <= 3 * sigma
        if keep.mean() < 0.5:
            return False
        gy, gx = np.mgrid[:h, :w]
        denom = coef[0] * gx + coef[1] * gy + coef[2]
        self.z_plane = np.where(denom > 1e-3, 1.0 / np.maximum(denom, 1e-3), np.nan).astype(np.float32)
        self.median_z = float(np.median(z[ys, xs]))
        err = np.abs(1.0 / (A[keep] @ coef) - 1.0 / inv[keep])
        self.fit_error_mm = float(np.median(err) * 1000)
        return True

    def evaluate(self, depth_m):
        d = np.where(depth_m > 0, depth_m - self.z_plane, np.nan).astype(np.float32)
        return cv2.warpPerspective(d, self.H, WARP_SIZE, flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))


HUE_VERIFY = 0.25  # min fraction of a cell's disc area that must have the colour for a SAM detection to count


def hue_masks(frame_bgr):
    """Pixels that look red / blue regardless of SAM (used only where depth says a disc fills the hole)."""
    h, s, v = cv2.split(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV))
    return {"blue": (h >= 85) & (h <= 130) & (s >= 80) & (v >= 30),
            "red": ((h <= 10) | (h >= 165)) & (s >= 110) & (v >= 40)}


def depth_regions(delta, front, behind):
    valid = ~np.isnan(delta)
    inband = valid & (delta >= -front) & (delta <= behind)
    return valid, inband, valid & ~inband


def classify_cells(masks, min_coverage, frame=None, delta=None, front=0.03, behind=0.025):
    """Return (grid, fill): grid[row][col] in {'empty','red','blue'} (row 0 is the top row);
    fill[row][col] is the fraction of a cell's valid depth pixels that lie in the disc layer (None without depth).

    With depth: pixels in front of / behind the board plane are removed from the SAM masks; a SAM detection only
    counts if the cell really has that colour (hue check); and where depth says something fills the hole, the hue
    alone is accepted as evidence, so a SAM dropout does not lose a disc.
    """
    valid = inband = hue = None
    if delta is not None:
        valid, inband, foreign = depth_regions(delta, front, behind)
        masks = {k: m & ~foreign for k, m in masks.items()}
        if frame is not None:
            hue = hue_masks(frame)
    grid, fills = [], []
    for r in range(ROWS):
        row, frow = [], []
        for c in range(COLS):
            sl = (slice(r * CELL, (r + 1) * CELL), slice(c * CELL, (c + 1) * CELL))
            cov = {k: float(m[sl][_DISC].mean()) for k, m in masks.items()}
            fill = None
            if valid is not None:
                v = valid[sl][_DISC]
                if v.mean() >= 0.3:
                    fill = float(inband[sl][_DISC].sum() / max(int(v.sum()), 1))
            if hue is not None:
                for k in cov:
                    colour = float(hue[k][sl][_DISC].mean())
                    if colour < HUE_VERIFY:       # SAM says disc, but the cell does not have that colour
                        cov[k] = 0.0
                    if fill is not None and fill >= 0.6:   # depth says the hole is filled: colour alone is evidence
                        cov[k] = max(cov[k], colour)
            best = max(cov, key=cov.get)
            row.append(best if cov[best] >= min_coverage else "empty")
            frow.append(fill)
        grid.append(row)
        fills.append(frow)
    return grid, fills


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
    def __init__(self, segmenter_factory, state, prompts, min_coverage, tracker, front=0.03, behind=0.025):
        super().__init__(daemon=True)
        self.factory, self.state, self.prompts = segmenter_factory, state, prompts
        self.min_coverage, self.tracker = min_coverage, tracker
        self.front, self.behind = front, behind
        self.delta = None      # latest depth-vs-board-plane map (None without depth)
        self.rejected = None   # SAM pixels thrown away by the depth filter, for the preview
        self.fills = None
        self.lock = threading.Lock()
        self.frame, self.frame_id = None, 0
        self.observed = None
        self.masks = None  # latest SAM masks {colour: bool array}, for the preview overlay
        self.stop_event = threading.Event()

    def submit(self, warped, delta=None):
        with self.lock:
            self.frame, self.delta, self.frame_id = warped, delta, self.frame_id + 1

    def run(self):
        self.state.set_status("loading SAM3 model")
        segmenter = self.factory()
        self.state.set_status("waiting for frames")
        last_id = 0
        while not self.stop_event.is_set():
            with self.lock:
                frame, delta, fid = self.frame, self.delta, self.frame_id
            if frame is None or fid == last_id:
                time.sleep(0.01)
                continue
            last_id = fid
            masks = segmenter.segment(frame, self.prompts)
            if delta is not None:
                foreign = depth_regions(delta, self.front, self.behind)[2]
                self.rejected = {k: m & foreign for k, m in masks.items()}
                self.masks = {k: m & ~foreign for k, m in masks.items()}
            else:
                self.rejected, self.masks = None, masks
            raw, self.fills = classify_cells(masks, self.min_coverage, frame, delta, self.front, self.behind)
            observed = self.tracker.update(raw, time.monotonic())
            errors, _ = validate(observed)
            self.observed = observed
            self.state.update(observed, errors, self.tracker.game_id)


# --------------------------------------------------------------------------- main
def annotate(warped, observed, masks=None, rejected=None, fills=None):
    view = warped.copy()
    if masks:  # tint SAM masks and outline them (red prompt -> red, blue prompt -> cyan)
        for colour, bgr in (("red", (0, 255, 255)), ("blue", (255, 255, 0))):
            m = masks.get(colour)
            if m is None or m.shape != view.shape[:2]:
                continue
            view[m] = (0.5 * view[m] + 0.5 * np.array(bgr)).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, cnts, -1, bgr, 2)
    if rejected:  # detected by SAM but not in the board plane (behind the board / in front of it): magenta
        for m in rejected.values():
            if m is not None and m.shape == view.shape[:2] and m.any():
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(view, cnts, -1, (255, 0, 255), 2)
    draw_grid(view)
    if fills:  # per-cell fraction of the hole that is filled according to depth
        for r in range(ROWS):
            for c in range(COLS):
                if fills[r][c] is not None:
                    cv2.putText(view, f"{fills[r][c]:.2f}", (c * CELL + 30, r * CELL + CELL - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
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
    ap.add_argument("--blue-prompt", default="blue disc, blue circle",
                    help="comma-separated prompts are combined (union); \"blue disc\" alone is flaky")
    ap.add_argument("--confidence", type=float, default=0.4, help="SAM3 detection threshold")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="fraction of a cell centre a mask must cover")
    ap.add_argument("--confirm-seconds", type=float, default=0.5, help="a new disc must be seen this long before it is added")
    ap.add_argument("--flip-seconds", type=float, default=5.0, help="a settled disc changes colour only after this long")
    ap.add_argument("--reset-seconds", type=float, default=5.0, help="no discs detected for this long = game restarted, clear the board")
    ap.add_argument("--no-depth", action="store_true", help="do not use the RealSense depth stream")
    ap.add_argument("--depth-behind", type=float, default=0.025,
                    help="metres behind the board's front face that still count as a disc (anything further is background)")
    ap.add_argument("--depth-front", type=float, default=0.03,
                    help="metres in front of the board's front face that still count as a disc (closer is a hand/arm)")
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
        camera = RealSenseCamera(args.width, args.height, args.fps, use_depth=not args.no_depth)
    try:
        if args.no_display and not args.calibration.exists():
            raise SystemExit("--no-display needs an existing calibration file")
        H = homography(load_or_calibrate(camera, args.calibration, args.recalibrate))

        gate = None
        if camera.has_depth and not args.no_depth:
            frames = []
            while len(frames) < 15:  # median of a few frames for a clean plane fit
                if camera.read() is not None and camera.depth is not None:
                    frames.append(camera.depth)
            gate = DepthGate(H, (frames[0].shape[1], frames[0].shape[0]))
            if gate.fit(frames):
                print(f"depth filter on: board plane at {gate.median_z:.2f} m (fit error {gate.fit_error_mm:.1f} mm); "
                      f"discs must be {args.depth_front * 100:g} cm in front to {args.depth_behind * 100:g} cm behind it")
            else:
                print("warning: could not fit the board plane from depth (is the board in view and is depth valid "
                      "on it?), running without the depth filter")
                gate = None
        elif not args.no_depth:
            print("no depth stream (video/webcam/stream source): depth filter off")

        state = BoardState()
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        app = make_app(state)
        threading.Thread(target=lambda: app.run(args.host, args.port, threaded=True), daemon=True).start()
        print(f"REST API: http://{args.host}:{args.port}/state")

        detector = Detector(lambda: Sam3Segmenter(args.confidence), state,
                            {"red": args.red_prompt, "blue": args.blue_prompt},
                            args.min_coverage,
                            BoardTracker(args.confirm_seconds, args.flip_seconds, args.reset_seconds),
                            args.depth_front, args.depth_behind)
        detector.start()

        show_masks = not args.hide_masks
        show_depth = False
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            warped = cv2.warpPerspective(frame, H, WARP_SIZE)
            delta = gate.evaluate(camera.depth) if gate is not None and camera.depth is not None else None
            detector.submit(warped, delta)
            if args.no_display:
                continue
            snap = state.preview()
            view = annotate(warped, snap["observed"], detector.masks if show_masks else None,
                            detector.rejected if show_masks else None, detector.fills if show_depth else None)
            label = snap["status"] if snap["status"] != "ok" else ("; ".join(snap["errors"]) or "valid")
            cv2.putText(view, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Connect Four", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                show_masks = not show_masks
            elif key == ord("d"):
                show_depth = not show_depth
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
