"""
Re-render a LIBERO dataset at a higher camera resolution.

The shipped libero_spatial demos store RGB at 128x128, which the dataloader then
bilinearly upsamples to 384 (the model input) -- so the encoders only ever see
128-worth of detail. This script replays each demo's recorded mujoco states in
the simulator and re-captures the two camera views at a chosen resolution,
copying everything else (actions, states, proprio) verbatim so the new dataset
is a drop-in higher-resolution replacement.

State->observation mapping (validated empirically, near-zero MAE on the wrist
camera): stored obs[j] corresponds to render(states[j+1]). The final frame is
obtained by stepping the last action from states[-1].

Usage:
    python rerender_dataset.py \
        --src-dir libero/libero/datasets/libero_spatial \
        --dst-dir libero/libero/datasets/libero_spatial_384 \
        --res 384
    # quick correctness/timing check (re-render@128 and diff vs stored, 1 file):
    python rerender_dataset.py --sanity --limit-files 1
"""

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

for _i, _finder in enumerate(sys.meta_path):
    if getattr(_finder, "__name__", "") == "_EditableFinder" and "libero" in getattr(
        _finder, "__module__", ""
    ):
        sys.meta_path.insert(0, sys.meta_path.pop(_i))
        break

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

AGENT_CAM = "agentview"
WRIST_CAM = "robot0_eye_in_hand"
# obs dataset names in the hdf5 (what the dataloader reads)
AGENT_KEY = "agentview_rgb"
WRIST_KEY = "eye_in_hand_rgb"
# obs subgroups that are images (re-rendered) vs copied verbatim
IMG_KEYS = {AGENT_KEY, WRIST_KEY}


def resolve_bddl(bddl_rel: str) -> str:
    path = os.path.join(
        get_libero_path("bddl_files"), bddl_rel.split("bddl_files/")[-1]
    )
    assert os.path.exists(path), f"bddl not found: {path}"
    return path


def render_demo(env, states, actions, res):
    """Return (agentview (T,res,res,3), wrist (T,res,res,3)) uint8."""
    T = states.shape[0]
    av = np.empty((T, res, res, 3), dtype=np.uint8)
    eh = np.empty((T, res, res, 3), dtype=np.uint8)
    for j in range(T):
        if j < T - 1:
            obs = env.regenerate_obs_from_state(states[j + 1])
        else:
            # post-final observation: set last state, apply last action
            env.set_state(states[j])
            env.sim.forward()
            obs, _, _, _ = env.step(actions[j])
        av[j] = obs[f"{AGENT_CAM}_image"]
        eh[j] = obs[f"{WRIST_CAM}_image"]
    return av, eh


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def process_file(src_path, dst_path, res, sanity):
    fin = h5py.File(src_path, "r")
    bddl_file = resolve_bddl(fin["data"].attrs["bddl_file_name"])
    demo_keys = sorted(fin["data"].keys(), key=lambda k: int(k.split("_")[-1]))

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=[AGENT_CAM, WRIST_CAM],
        camera_heights=res,
        camera_widths=res,
        use_camera_obs=True,
    )
    env.seed(0)
    env.reset()

    if sanity:
        # re-render demo_0 and diff against stored (only valid when res==128)
        d = fin["data"]["demo_0"]
        states = d["states"][()]
        actions = d["actions"][()]
        av, eh = render_demo(env, states, actions, res)
        sa = d["obs"][AGENT_KEY][()]
        se = d["obs"][WRIST_KEY][()]
        if av.shape == sa.shape:
            print(
                f"  [sanity] agentview MAE={np.abs(av.astype(int)-sa).mean():.3f} "
                f"wrist MAE={np.abs(eh.astype(int)-se).mean():.3f}"
            )
        else:
            print(f"  [sanity] shape differs (res={res}); skipping pixel diff")
        env.close()
        fin.close()
        return 0

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fout = h5py.File(dst_path, "w")
    gin = fin["data"]
    gout = fout.create_group("data")
    copy_attrs(gin, gout)

    total = 0
    for dk in demo_keys:
        din = gin[dk]
        states = din["states"][()]
        actions = din["actions"][()]
        av, eh = render_demo(env, states, actions, res)

        dout = gout.create_group(dk)
        copy_attrs(din, dout)
        # copy non-image datasets verbatim
        for k in din.keys():
            if k == "obs":
                continue
            dout.create_dataset(k, data=din[k][()])
        oin = din["obs"]
        oout = dout.create_group("obs")
        copy_attrs(oin, oout)
        for k in oin.keys():
            if k in IMG_KEYS:
                continue
            oout.create_dataset(k, data=oin[k][()])
        oout.create_dataset(AGENT_KEY, data=av, compression="gzip", compression_opts=4)
        oout.create_dataset(WRIST_KEY, data=eh, compression="gzip", compression_opts=4)
        total += states.shape[0]

    env.close()
    fout.close()
    fin.close()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="libero/libero/datasets/libero_spatial")
    ap.add_argument("--dst-dir", default="libero/libero/datasets/libero_spatial_384")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="re-render demo_0 and diff vs stored (use --res 128)",
    )
    ap.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="process only the first N task files (0 = all)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a file if its dst exists with the same demo count",
    )
    args = ap.parse_args()

    src_paths = sorted(glob.glob(os.path.join(args.src_dir, "*.hdf5")))
    if args.limit_files:
        src_paths = src_paths[: args.limit_files]
    if not src_paths:
        raise FileNotFoundError(f"no .hdf5 in {args.src_dir}")

    print(f"{len(src_paths)} file(s), res={args.res}, sanity={args.sanity}")
    grand = 0
    for i, sp in enumerate(src_paths):
        name = os.path.basename(sp)
        dp = os.path.join(args.dst_dir, name)
        if args.skip_existing and not args.sanity and os.path.exists(dp):
            try:
                with h5py.File(dp, "r") as fc, h5py.File(sp, "r") as fs:
                    if len(fc["data"].keys()) == len(fs["data"].keys()):
                        print(f"[{i+1}/{len(src_paths)}] {name}: skip (exists)")
                        continue
            except Exception:
                pass  # unreadable/partial -> re-render
        t0 = time.perf_counter()
        n = process_file(sp, dp, args.res, args.sanity)
        dt = time.perf_counter() - t0
        grand += n
        extra = "" if args.sanity else f"-> {dp}"
        print(f"[{i+1}/{len(src_paths)}] {name}: {n} steps in {dt:.1f}s {extra}")
    if not args.sanity:
        print(f"Done. {grand} steps re-rendered at {args.res}x{args.res}.")


if __name__ == "__main__":
    main()
