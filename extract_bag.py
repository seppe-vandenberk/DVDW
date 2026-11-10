#!/usr/bin/env python3
"""Extract the colour stream of a RealSense .bag recording to an mp4 file."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: <bag>.mp4")
    args = ap.parse_args()
    out_path = args.output or args.bag.with_suffix(".mp4")

    config = rs.config()
    rs.config.enable_device_from_file(config, str(args.bag), repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)  # don't drop frames
    color = next(s for s in profile.get_streams() if s.stream_type() == rs.stream.color).as_video_stream_profile()
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), color.fps(),
                             (color.width(), color.height()))
    n = 0
    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(5000)
            if not ok:
                break
            frame = frames.get_color_frame()
            if not frame:
                continue
            rgb = np.asanyarray(frame.get_data())
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if frame.profile.format() == rs.format.rgb8 else rgb)
            n += 1
            if n % 500 == 0:
                print(f"{n} frames, {playback.get_position() / 1e9:.0f}s", flush=True)
            if playback.current_status() == rs.playback_status.stopped:
                break
    finally:
        writer.release()
        pipeline.stop()
    print(f"wrote {n} frames to {out_path}")


if __name__ == "__main__":
    main()
