"""Dump one recorded LIBERO demo as a video, rendered exactly the way the model
sees it during training, so it can be eyeballed against eval-time rollouts.

The frames match eval_libero.annotate_frame (180-degree rotation + goal banner),
so the output MP4 is directly comparable to the videos saved by
`python eval_libero.py --save_video`.

Example:
    python visualize_libero_demo.py --demo demo_5
    python visualize_libero_demo.py --demo demo_5 --camera eye_in_hand_rgb
"""

import argparse
import glob
import json
import os
import textwrap

import cv2
import h5py
import imageio
import numpy as np

# Wrist cameras are already upright; only the third-person view is upside down
# (must match libero_dataloader.WRIST_CAMERAS / eval_libero.preprocess_obs).
WRIST_CAMERAS = ("eye_in_hand_rgb",)


def annotate_frame(
    frame: np.ndarray,
    instruction: str,
    step: int,
    max_steps: int,
    rotate_180: bool,
    out_size: int = 512,
) -> np.ndarray:
    """Rotate (to match the model's input orientation), upscale, and draw the
    goal + step counter. Mirrors eval_libero.annotate_frame."""
    if rotate_180:
        frame = np.ascontiguousarray(np.rot90(frame, k=2))
    frame = cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_h = 22
    lines = textwrap.wrap(f"Goal: {instruction}", width=48) or ["Goal:"]
    banner_h = line_h * len(lines) + 12
    banner = np.zeros((banner_h, out_size, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            banner,
            line,
            (8, line_h * (i + 1)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"[recorded demo]  step {step}/{max_steps}",
        (8, 22),
        font,
        font_scale,
        (0, 200, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.concatenate([frame, banner], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="libero/libero/datasets/libero_spatial_384",
        help="Directory containing the *.hdf5 demo file(s)",
    )
    parser.add_argument(
        "--hdf5",
        default=None,
        help="Specific hdf5 file (defaults to the first one found in dataset_dir)",
    )
    parser.add_argument(
        "--demo",
        default="demo_5",
        help="Demo key to export (demo_0..4 are the val split, 5+ are train)",
    )
    parser.add_argument("--camera", default="agentview_rgb")
    parser.add_argument("--fps", type=int, default=20, help="LIBERO control_freq is 20")
    parser.add_argument("--out_dir", default="dataset_videos")
    args = parser.parse_args()

    if args.hdf5 is not None:
        hdf5_path = args.hdf5
    else:
        paths = sorted(glob.glob(os.path.join(args.dataset_dir, "*.hdf5")))
        if not paths:
            raise FileNotFoundError(f"No .hdf5 files in {args.dataset_dir!r}")
        hdf5_path = paths[0]

    rotate_180 = args.camera not in WRIST_CAMERAS

    with h5py.File(hdf5_path, "r") as f:
        instruction = json.loads(f["data"].attrs["problem_info"])[
            "language_instruction"
        ]
        if args.demo not in f["data"]:
            raise KeyError(
                f"{args.demo!r} not in file; available: {sorted(f['data'].keys())[:8]}..."
            )
        images = f["data"][args.demo]["obs"][args.camera][:]  # (T, H, W, 3) uint8

    n = images.shape[0]
    print(f"File:        {hdf5_path}")
    print(f"Demo:        {args.demo}  ({n} steps, ~{n / args.fps:.1f}s @ {args.fps}Hz)")
    print(f"Camera:      {args.camera}  (rotate_180={rotate_180})")
    print(f"Instruction: {instruction}")

    frames = [
        annotate_frame(images[t], instruction, t + 1, n, rotate_180) for t in range(n)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    base = f"{args.demo}_{args.camera}"
    video_path = os.path.join(args.out_dir, f"{base}.mp4")
    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=None)
    print(f"Saved video: {video_path}")

    # Also dump a first/mid/last still-frame strip for a quick glance.
    picks = [0, n // 2, n - 1]
    strip = np.concatenate([frames[i] for i in picks], axis=1)
    strip_path = os.path.join(args.out_dir, f"{base}_strip.png")
    imageio.imwrite(strip_path, strip)
    print(f"Saved strip: {strip_path}  (steps {[p + 1 for p in picks]})")


if __name__ == "__main__":
    main()
