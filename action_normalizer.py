"""OpenVLA-style action normalization.

Continuous actions are normalized per-dimension to [-1, 1] using the 1st and
99th percentiles (q01/q99) of the training distribution, then clipped. Using
percentiles (instead of min/max) keeps outliers from compressing the useful
range into a few bins. A per-dimension mask leaves selected dims unnormalized
(the gripper is binary and should not be rescaled).

The same statistics are used at train time (normalize -> tokenize) and at
inference time (detokenize -> denormalize), so they are saved alongside the
checkpoint and reloaded for evaluation.
"""

import glob
import json
import os

import h5py
import numpy as np

EPS = 1e-8


class ActionNormalizer:
    def __init__(self, q01, q99, mask):
        """
        q01, q99: (action_dim,) per-dim 1st/99th percentile of raw actions.
        mask:     (action_dim,) bool; True = normalize this dim, False = leave raw.
        """
        self.q01 = np.asarray(q01, dtype=np.float32)
        self.q99 = np.asarray(q99, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=bool)
        # Guard against zero-width ranges (constant dims) to avoid div-by-zero.
        self._denom = np.where(
            np.abs(self.q99 - self.q01) > EPS, self.q99 - self.q01, 1.0
        )

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Raw action -> [-1, 1] (masked dims pass through unchanged)."""
        action = np.asarray(action, dtype=np.float32)
        normed = np.clip(2 * (action - self.q01) / self._denom - 1, -1.0, 1.0)
        return np.where(self.mask, normed, action)

    def denormalize(self, normed: np.ndarray) -> np.ndarray:
        """[-1, 1] -> raw action (masked dims pass through unchanged)."""
        normed = np.asarray(normed, dtype=np.float32)
        raw = (normed + 1) / 2 * self._denom + self.q01
        return np.where(self.mask, raw, normed)

    def to_dict(self) -> dict:
        return {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "mask": self.mask.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionNormalizer":
        return cls(d["q01"], d["q99"], d["mask"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ActionNormalizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return f"ActionNormalizer(q01={self.q01}, q99={self.q99}, mask={self.mask})"


def compute_action_stats(
    dataset_dir: str | list[str],
    split: str = "train",
    val_demos: int = 5,
    noop_thresh: float = 1e-4,
    gripper_dim: int = 6,
) -> ActionNormalizer:
    """Compute q01/q99 over the (no-op filtered) split, matching the dataloader.

    The gripper dimension is excluded from normalization (mask=False).
    Supports single directory or list of directories.
    """
    if isinstance(dataset_dir, str):
        dataset_dirs = [dataset_dir]
    else:
        dataset_dirs = list(dataset_dir)

    collected = []
    for dir_path in dataset_dirs:
        hdf5_paths = sorted(glob.glob(os.path.join(dir_path, "*.hdf5")))
        if not hdf5_paths:
            print(f"Warning: No .hdf5 files found in {dir_path!r}")
            continue

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                all_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                use_keys = (
                    all_keys[val_demos:] if split == "train" else all_keys[:val_demos]
                )
                for demo_key in use_keys:
                    actions = f["data"][demo_key]["actions"][:]  # (T, 7)
                    prev_gripper = None
                    for step in range(actions.shape[0]):
                        a = actions[step]
                        moving = np.linalg.norm(a[:6]) >= noop_thresh
                        gripper_change = prev_gripper is None or a[6] != prev_gripper
                        prev_gripper = a[6]
                        if not moving and not gripper_change:
                            continue
                        collected.append(a)

    if not collected:
        raise FileNotFoundError(f"No actions collected from {dataset_dirs}")

    acts = np.asarray(collected, dtype=np.float64)  # (N, action_dim)
    q01 = np.percentile(acts, 1, axis=0)
    q99 = np.percentile(acts, 99, axis=0)
    mask = np.ones(acts.shape[1], dtype=bool)
    mask[gripper_dim] = False  # leave gripper unnormalized
    return ActionNormalizer(q01, q99, mask)
