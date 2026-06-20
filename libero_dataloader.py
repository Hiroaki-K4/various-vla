import glob
import json
import os
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from action_tokenizer import ActionTokenizer

IMAGE_SIZE = (384, 384)
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


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
        dataset_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        action_tokenizer: ActionTokenizer,
        split: str = "train",
        val_demos: int = 5,
        camera: str = "agentview_rgb",
    ):
        assert split in (
            "train",
            "val",
        ), f"split must be 'train' or 'val', got {split!r}"
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.camera = camera

        # Build flat index: list of (hdf5_path, demo_key, step_idx, instruction)
        self.samples: list[tuple[str, str, int, str]] = []

        hdf5_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.hdf5")))
        if not hdf5_paths:
            raise FileNotFoundError(f"No .hdf5 files found in {dataset_dir!r}")

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                instruction = json.loads(f["data"].attrs["problem_info"])[
                    "language_instruction"
                ]
                all_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))

                use_keys = (
                    all_keys[val_demos:] if split == "train" else all_keys[:val_demos]
                )

                for demo_key in use_keys:
                    n_steps = f["data"][demo_key]["actions"].shape[0]
                    for step in range(n_steps):
                        self.samples.append((hdf5_path, demo_key, step, instruction))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        hdf5_path, demo_key, step, instruction = self.samples[idx]

        # Open per call — safe for DataLoader multiprocessing (h5py handles
        # are not picklable, so we cannot hold one open across worker forks).
        with h5py.File(hdf5_path, "r") as f:
            image_np = f["data"][demo_key]["obs"][self.camera][step]  # (H, W, 3) uint8
            action_np = f["data"][demo_key]["actions"][step].astype(np.float32)  # (7,)

        # (H, W, 3) uint8  →  (3, 384, 384) float32 in [0, 1]
        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image = F.interpolate(
            image.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False
        ).squeeze(0)

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
        }


def _collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    images = torch.stack([b["image"] for b in batch])
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list, masks_list, labels_list = [], [], []
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
    }


def get_libero_dataloader(
    dataset_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    action_tokenizer: ActionTokenizer,
    batch_size: int = 8,
    split: str = "train",
    num_workers: int = 0,
    val_demos: int = 5,
    camera: str = "agentview_rgb",
) -> DataLoader:
    """
    Args:
        dataset_dir: path to a libero task-suite directory, e.g.
                     "libero/libero/datasets/libero_spatial"
        val_demos:   first N demos per HDF5 file reserved for validation.
    """
    dataset = LiberoDataset(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        split=split,
        val_demos=val_demos,
        camera=camera,
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collate = partial(_collate_fn, pad_token_id=pad_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
