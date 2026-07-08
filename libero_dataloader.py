import glob
import json
import os
import random
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from action_normalizer import ActionNormalizer
from action_tokenizer import ActionTokenizer

IMAGE_SIZE = (384, 384)
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

# Threshold below which an action's translation/rotation is treated as "no motion".
# Matches the OpenVLA paper's `is_noop`: L2 norm of the non-gripper dims < 1e-4.
NOOP_THRESH = 1e-4

# Wrist camera(s) are already upright; only the third-person view is upside down
# on our setup, so we rotate everything except these by 180 degrees.
WRIST_CAMERAS = ("eye_in_hand_rgb",)


class LiberoDataset(Dataset):
    """
    Reads libero HDF5 demo files.  Each sample is one (observation, action) step.

    Directory layout expected:
        dataset_dir/
            <task_name>_demo.hdf5
            ...

    Each HDF5 file has the structure:
        data/
          demo_0/
            obs/agentview_rgb   (T, 128, 128, 3) uint8
            obs/eye_in_hand_rgb (T, 128, 128, 3) uint8
            actions             (T, 7)  float64
          demo_1/ ...
        data.attrs["problem_info"]  JSON with "language_instruction"
    """

    def __init__(
        self,
        dataset_dir: str | list[str],
        tokenizer: PreTrainedTokenizerBase,
        action_tokenizer: ActionTokenizer,
        split: str = "train",
        val_demos: int = 5,
        cameras: tuple[str, ...] = ("agentview_rgb",),
        action_normalizer: ActionNormalizer | None = None,
        val_step_ratio: float = 1.0,
        image_aug: bool = False,
    ):
        assert split in (
            "train",
            "val",
        ), f"split must be 'train' or 'val', got {split!r}"
        assert len(cameras) >= 1, "at least one camera is required"
        assert (
            0 < val_step_ratio <= 1.0
        ), f"val_step_ratio must be in (0, 1], got {val_step_ratio}"
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.action_normalizer = action_normalizer
        self.split = split
        self.val_step_ratio = val_step_ratio
        # One or more camera views (e.g. third-person + wrist). Each view is
        # encoded independently and its patch tokens are concatenated downstream.
        self.cameras = tuple(cameras)
        # LIBERO third-person images come in upside down on our hardware; rotate
        # them 180 degrees. Wrist images are already upright, so they are left
        # as-is. Rotation is decided per view.
        self.rotate_180 = tuple(cam not in WRIST_CAMERAS for cam in self.cameras)

        # Build flat index: list of (hdf5_path, demo_key, step_idx, instruction, task_name)
        self.samples: list[tuple[str, str, int, str, str]] = []

        # Support both single directory and list of directories
        if isinstance(dataset_dir, str):
            dataset_dirs = [dataset_dir]
        else:
            dataset_dirs = list(dataset_dir)

        for dir_path in dataset_dirs:
            # Extract task name from directory path (e.g., "libero_spatial" from path)
            task_name = os.path.basename(dir_path.rstrip("/"))

            hdf5_paths = sorted(glob.glob(os.path.join(dir_path, "*.hdf5")))
            if not hdf5_paths:
                print(f"Warning: No .hdf5 files found in {dir_path!r}")
                continue

            for hdf5_path in hdf5_paths:
                with h5py.File(hdf5_path, "r") as f:
                    instruction = json.loads(f["data"].attrs["problem_info"])[
                        "language_instruction"
                    ]
                    all_keys = sorted(
                        f["data"].keys(), key=lambda k: int(k.split("_")[-1])
                    )

                    use_keys = (
                        all_keys[val_demos:]
                        if split == "train"
                        else all_keys[:val_demos]
                    )

                    for demo_key in use_keys:
                        actions = f["data"][demo_key]["actions"][:]  # (T, 7)
                        n_steps = actions.shape[0]
                        prev_gripper = None
                        valid_steps = []
                        for step in range(n_steps):
                            a = actions[step]
                            # a[:6] = translation (3) + rotation (3), a[6] = gripper
                            moving = np.linalg.norm(a[:6]) >= NOOP_THRESH
                            gripper_change = (
                                prev_gripper is None or a[6] != prev_gripper
                            )
                            prev_gripper = a[6]
                            # Drop "no-op" steps: no motion AND no gripper state change.
                            if not moving and not gripper_change:
                                continue
                            valid_steps.append(step)

                        # For val split, optionally subsample steps
                        if split == "val" and self.val_step_ratio < 1.0:
                            np.random.seed(
                                hash(demo_key) % (2**31)
                            )  # Deterministic sampling
                            n_samples = max(
                                1, int(len(valid_steps) * self.val_step_ratio)
                            )
                            valid_steps = list(
                                np.random.choice(
                                    valid_steps, size=n_samples, replace=False
                                )
                            )

                        for step in valid_steps:
                            self.samples.append(
                                (hdf5_path, demo_key, step, instruction, task_name)
                            )

        # Define augmentation
        self.image_aug = image_aug and (split == "train")
        if self.image_aug:
            self.aug_transform = T.Compose(
                [
                    T.RandomResizedCrop(
                        size=IMAGE_SIZE, scale=(0.9, 0.9), ratio=(1.0, 1.0)
                    ),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                ]
            )
        else:
            self.aug_transform = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        hdf5_path, demo_key, step, instruction, task_name = self.samples[idx]

        # Open per call — safe for DataLoader multiprocessing (h5py handles
        # are not picklable, so we cannot hold one open across worker forks).
        view_imgs = []
        with h5py.File(hdf5_path, "r") as f:
            obs = f["data"][demo_key]["obs"]
            seed = random.randint(0, 2**31 - 1) if self.image_aug else None
            for i, (cam, rotate) in enumerate(zip(self.cameras, self.rotate_180)):
                image_np = obs[cam][step]  # (H, W, 3) uint8
                if rotate:
                    image_np = np.rot90(image_np, k=2)  # flip upside down
                # (H, W, 3) uint8  →  (3, 384, 384) float32 in [0, 1]
                img = (
                    torch.from_numpy(np.ascontiguousarray(image_np))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=IMAGE_SIZE,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                if self.aug_transform is not None:
                    torch.manual_seed(seed + i)
                    np.random.seed(seed + i)
                    img = self.aug_transform(img)

                view_imgs.append(img)
            action_np = f["data"][demo_key]["actions"][step].astype(np.float32)  # (7,)

        # (V, 3, 384, 384) — one entry per camera view.
        image = torch.stack(view_imgs, dim=0)

        # Normalize raw action to [-1, 1] (q01/q99, gripper left raw) so the 256
        # bins cover the useful range instead of collapsing into the center.
        if self.action_normalizer is not None:
            action_np = self.action_normalizer.normalize(action_np)

        # Tokenize continuous action (7,) → token ids (7,)
        action_token_ids = torch.tensor(
            self.action_tokenizer.tokenize(action_np), dtype=torch.long
        )

        # Prompt → token ids
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"].squeeze(0)

        input_ids = torch.cat([prompt_ids, action_token_ids])
        # Mask prompt tokens from loss; supervise only on action tokens.
        labels = torch.cat([torch.full_like(prompt_ids, -100), action_token_ids])
        attention_mask = torch.ones_like(input_ids)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_name": task_name,
        }


def _collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    images = torch.stack([b["image"] for b in batch])
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list, masks_list, labels_list = [], [], []
    task_names = [b["task_name"] for b in batch]

    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids_list.append(F.pad(b["input_ids"], (0, pad_len), value=pad_token_id))
        masks_list.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
        labels_list.append(F.pad(b["labels"], (0, pad_len), value=-100))

    return {
        "image": images,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(masks_list),
        "labels": torch.stack(labels_list),
        "task_names": task_names,
    }


def get_libero_dataloader(
    dataset_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    action_tokenizer: ActionTokenizer,
    batch_size: int = 8,
    split: str = "train",
    num_workers: int = 0,
    val_demos: int = 5,
    cameras: tuple[str, ...] = ("agentview_rgb",),
    action_normalizer: ActionNormalizer | None = None,
    seed: int | None = None,
    val_step_ratio: float = 1.0,
    image_aug: bool = True,
) -> DataLoader:
    """
    Args:
        dataset_dir: path to a libero task-suite directory, e.g.
                     "libero/libero/datasets/libero_spatial"
        val_demos:   first N demos per HDF5 file reserved for validation.
        val_step_ratio: for val split, ratio of steps to sample (0 < ratio <= 1.0).
        seed:        if set, makes shuffling and worker RNG reproducible.
    """
    dataset = LiberoDataset(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        split=split,
        val_demos=val_demos,
        cameras=cameras,
        action_normalizer=action_normalizer,
        val_step_ratio=val_step_ratio,
        image_aug=image_aug,
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collate = partial(_collate_fn, pad_token_id=pad_id)

    # Reproducible shuffling and per-worker RNG when a seed is provided.
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def worker_init_fn(worker_id: int) -> None:
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
