import io
import time
import warnings

import datasets
import numpy as np
import torch
import webdataset.compat as _wds_compat
from datasets import Features, IterableDataset, Sequence, Value
from PIL import Image
from torch.utils.data import DataLoader

from action_tokenizer import ActionTokenizer

# Silence the noisy `WebDataset(shardshuffle=...) is None` warning emitted from
# inside `webdataset.compat` for every shard in every worker. We don't control
# the call site (it lives inside the OpenX-Embodiment loader script), so we
# just filter it here.
warnings.filterwarnings(
    "ignore",
    message=r".*WebDataset\(shardshuffle=\.\.\.\) is None.*",
    category=UserWarning,
)


# Disable webdataset's `check_empty` filter. By default it raises
# `ValueError: No samples found in dataset; perhaps you have fewer shards than
# workers.` whenever a single tar shard yields zero raw samples. With multi-
# worker streaming + sharded HF datasets, individual shards can legitimately be
# empty for a given worker (e.g. when the worker's slice of an n_shards=4
# sub-dataset happens to be a tar that fails to decode any sample), and the
# OpenX-Embodiment loader script we depend on constructs `WebDataset(...)`
# without exposing `empty_check`. The simplest fix is to make `check_empty` a
# passthrough.
def _check_empty_passthrough(source):
    yield from source


_wds_compat.check_empty = _check_empty_passthrough

IMAGE_SIZE = (384, 384)
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

# Explicit output schema for `chunk_episodes`. Providing this to `.map()` lets
# `interleave_datasets` skip its `_resolve_features()` step, which otherwise
# tries to download one tar shard from every sub-dataset just to peek at the
# schema (a curl/network failure on any of them aborts everything).
OUTPUT_FEATURES = Features(
    {
        "instruction": Value("string"),
        "image": Value("binary"),
        "action": Sequence(Value("float32"), length=7),
    }
)


DATASETS = [
    "fractal20220817_data",
    "kuka",
    "bridge",
    # "taco_play",
    # "jaco_play",
    # "berkeley_cable_routing",
    # "roboturk",
    # "nyu_door_opening_surprising_effectiveness",
    # "viola",
    # "berkeley_autolab_ur5",
    # "toto",
    # "language_table",
    # "columbia_cairlab_pusht_real",
    # "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    # "nyu_rot_dataset_converted_externally_to_rlds",
    # "stanford_hydra_dataset_converted_externally_to_rlds",
    # "austin_buds_dataset_converted_externally_to_rlds",
    # "nyu_franka_play_dataset_converted_externally_to_rlds",
    # "maniskill_dataset_converted_externally_to_rlds",
    # "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    # "ucsd_kitchen_dataset_converted_externally_to_rlds",
    # "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    # "austin_sailor_dataset_converted_externally_to_rlds",
    # "austin_sirius_dataset_converted_externally_to_rlds",
    # "bc_z",
    # "usc_cloth_sim_converted_externally_to_rlds",
    # "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    # "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    # "utokyo_saytap_converted_externally_to_rlds",
    # "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    # "utokyo_xarm_bimanual_converted_externally_to_rlds",
    # "robo_net",
    # "berkeley_mvp_converted_externally_to_rlds",
    # "berkeley_rpt_converted_externally_to_rlds",
    # "kaist_nonprehensile_converted_externally_to_rlds",
    # "stanford_mask_vit_converted_externally_to_rlds",
    # "tokyo_u_lsmo_converted_externally_to_rlds",
    # "dlr_sara_pour_converted_externally_to_rlds",
    # "dlr_sara_grid_clamp_converted_externally_to_rlds",
    # "dlr_edan_shared_control_converted_externally_to_rlds",
    # "asu_table_top_converted_externally_to_rlds",
    # "stanford_robocook_converted_externally_to_rlds",
    # "eth_agent_affordances",
    # "imperialcollege_sawyer_wrist_cam",
    # "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    # "uiuc_d3field",
    # "utaustin_mutex",
    # "berkeley_fanuc_manipulation",
    # "cmu_play_fusion",
    # "cmu_stretch",
    # "berkeley_gnm_recon",
    # "berkeley_gnm_cory_hall",
    # "berkeley_gnm_sac_son",
]


IMAGE_SIZE = (384, 384)


def get_dataloader(
    tokenizer,
    action_tokenizer: ActionTokenizer,
    batch_size: int = 8,
    split: str = "train",
    num_workers: int = 0,
    val_size: int = 256,
    min_shards: int | None = None,
):
    """
    `jxu124/OpenX-Embodiment` provides only a `train` split, so we carve a
    validation set out of it by taking the first `val_size` examples (per
    sub-dataset) as val and skipping them for train.

    `min_shards` filters out sub-datasets whose `n_shards` is below the
    threshold. Many OpenX sub-datasets ship as a single tar file (n_shards=1),
    and webdataset throws `ValueError: No samples found in dataset; perhaps
    you have fewer shards than workers.` when a DataLoader worker is assigned
    zero shards. Defaults to `max(num_workers, 1)` so we automatically drop
    any sub-dataset that can't supply a shard to every worker.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    if min_shards is None:
        min_shards = max(num_workers, 1)

    ds_list = []
    for name in DATASETS:
        ds = _load_openx_with_retry(name)
        if ds is None:
            # Sub-dataset is unreachable right now; skip it for this run
            # rather than aborting the whole training job.
            continue

        sub_shards = getattr(ds, "n_shards", 1) or 1
        if sub_shards < min_shards:
            print(
                f"[dataloader] skipping {name!r} (n_shards={sub_shards} < "
                f"min_shards={min_shards})"
            )
            continue

        def chunk_episodes(examples):
            out_instr, out_img, out_act = [], [], []
            episodes = examples.get("data.pickle", examples.get("steps", []))
            for episode in episodes:
                for step in episode["steps"]:
                    obs = step.get("observation", {})
                    image_field = obs.get("image", None)
                    if isinstance(image_field, dict):
                        image_bytes = image_field.get("bytes", None)
                    elif isinstance(image_field, (bytes, bytearray)):
                        image_bytes = bytes(image_field)
                    else:
                        image_bytes = None
                    if image_bytes is None:
                        continue

                    instruction = obs.get("natural_language_instruction", "")
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode("utf-8", errors="ignore")

                    act_vec = _action_to_vec(step.get("action"))
                    if act_vec is None:
                        continue

                    out_instr.append(instruction)
                    out_img.append(image_bytes)
                    out_act.append(act_vec.tolist())

            return {
                "instruction": out_instr,
                "image": out_img,
                "action": out_act,
            }

        ds = ds.map(
            chunk_episodes,
            batched=True,
            remove_columns=ds.column_names,
            features=OUTPUT_FEATURES,
        )

        if split == "val":
            ds = ds.take(val_size)
        else:  # "train"
            ds = ds.skip(val_size)

        ds_list.append(ds)

    if not ds_list:
        raise RuntimeError(
            "All OpenX-Embodiment sub-datasets failed to load. "
            "Check your network connection to huggingface.co."
        )

    combined_ds = datasets.interleave_datasets(ds_list, seed=42)

    def _collate(batch):
        return collate_fn(batch, tokenizer, action_tokenizer)

    # HuggingFace caps `num_workers` at `dataset.n_shards`. Because
    # `_make_resilient` wraps each sub-dataset via `IterableDataset.from_generator`
    # (which is single-shard), the interleaved dataset ends up with n_shards=1
    # and HF prints a noisy "Too many dataloader workers" warning while silently
    # downgrading to 1 worker. We cap here to avoid the warning and make the
    # actual worker count explicit.
    n_shards = getattr(combined_ds, "n_shards", 1) or 1
    effective_workers = min(num_workers, n_shards)
    if effective_workers != num_workers:
        print(
            f"[dataloader] capping num_workers {num_workers} -> {effective_workers} "
            f"(dataset n_shards={n_shards})"
        )

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=effective_workers,
        pin_memory=True,
    )
    if effective_workers > 0:
        # Larger prefetch helps when only a few workers are available: each
        # worker buffers more batches so the GPU loop doesn't block on the
        # next HF curl.
        loader_kwargs.update(
            prefetch_factor=8,
            persistent_workers=True,
        )

    return DataLoader(combined_ds, **loader_kwargs)


def _decode_image(image_bytes, image_size=IMAGE_SIZE):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(image_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


# Per-component fallback keys across OpenX sub-datasets. The schema for
# `step['action']` is *not* unified — some datasets give RT-1 style dicts
# (`world_vector` / `rotation_delta` / `gripper_closedness_action`), some use
# alternative names (`gripper`, `open_gripper`, ...), and a handful expose the
# action as a flat 7-dim tensor directly. We try a list of candidates per slot
# and skip the step if none of them are present.
_WORLD_VECTOR_KEYS = ("world_vector", "base_displacement_vector", "linear_velocity")
_ROTATION_KEYS = (
    "rotation_delta",
    "base_displacement_vertical_rotation",
    "angular_velocity",
)
_GRIPPER_KEYS = (
    "gripper_closedness_action",
    "gripper",
    "open_gripper",
    "grasp",
    "gripper_action",
)


def _first_present(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _action_to_vec(action) -> np.ndarray | None:
    """
    Best-effort conversion of an OpenX `step['action']` into a 7-dim vector
    `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. Returns None if the action
    cannot be mapped (caller should skip the step).
    """
    if action is None:
        return None

    # Some datasets already give a flat numeric action.
    if not isinstance(action, dict):
        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size < 7:
            return None
        return arr[:7]

    wv_raw = _first_present(action, _WORLD_VECTOR_KEYS)
    rot_raw = _first_present(action, _ROTATION_KEYS)
    grip_raw = _first_present(action, _GRIPPER_KEYS)
    if wv_raw is None or rot_raw is None or grip_raw is None:
        return None

    try:
        wv = np.asarray(wv_raw, dtype=np.float32).reshape(-1)
        rot = np.asarray(rot_raw, dtype=np.float32).reshape(-1)
        grip = np.asarray(grip_raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None

    if wv.size < 3 or rot.size < 3 or grip.size < 1:
        return None

    return np.concatenate([wv[:3], rot[:3], grip[:1]], axis=0)  # (7,)


def _load_openx_with_retry(
    name: str,
    max_retries: int = 5,
    backoff: float = 2.0,
):
    """
    Call `datasets.load_dataset("jxu124/OpenX-Embodiment", name, streaming=True)`
    with retry-on-network-error.

    The OpenX-Embodiment loader script peeks at one tar shard from inside
    `_split_generators` (to infer the schema), so transient HF `curl` failures
    surface here — before iteration even starts — and `_make_resilient` can't
    catch them. We retry a few times with exponential backoff, and return
    `None` if the sub-dataset is still unreachable so the caller can skip it.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return datasets.load_dataset(
                "jxu124/OpenX-Embodiment",
                name,
                streaming=True,
                split="train",
                trust_remote_code=True,
            )
        except (OSError, IOError) as e:  # noqa: UP024
            if attempt == max_retries:
                print(
                    f"[dataloader] sub-dataset {name!r} unreachable after "
                    f"{max_retries} load_dataset retries: {e}. Skipping."
                )
                return None
            delay = backoff**attempt
            print(
                f"[dataloader] sub-dataset {name!r} load_dataset failed "
                f"(attempt {attempt}/{max_retries}, retrying in {delay:.1f}s): {e}"
            )
            time.sleep(delay)
    return None


def _make_resilient(
    ds: IterableDataset, name: str, max_retries: int = 5
) -> IterableDataset:
    """
    Wrap a streaming dataset so that transient curl/network failures (e.g.
    `OSError: ... exit 92 (read)` — partial tar download from HF) don't kill
    the sub-dataset for the rest of the epoch.

    On error we re-create the iterator (which lets `datasets`/`webdataset`
    move past the bad shard on its next attempt) up to `max_retries` times
    before giving up. `interleave_datasets` keeps cycling through the other
    sub-datasets in the meantime.
    """

    def gen():
        retries = 0
        while True:
            try:
                for ex in iter(ds):
                    yield ex
                return  # iterator exhausted cleanly
            except (OSError, IOError) as e:  # noqa: UP024 (IOError kept for clarity)
                retries += 1
                if retries > max_retries:
                    print(
                        f"[dataloader] sub-dataset {name!r} giving up after "
                        f"{max_retries} retries: {e}"
                    )
                    return
                print(
                    f"[dataloader] sub-dataset {name!r} hit shard error "
                    f"(retry {retries}/{max_retries}): {e}"
                )
                # Loop around and rebuild the iterator from `ds`; HF's
                # streaming layer will re-issue curl for the next shard.

    return IterableDataset.from_generator(gen, features=OUTPUT_FEATURES)


def collate_fn(batch, tokenizer, action_tokenizer: ActionTokenizer):
    """
    Each sample = 1 step (image + instruction + action).

    For each sample,
        prompt_ids = tokenizer(PROMPT_TEMPLATE.format(instruction=...))
        action_ids = action_tokenizer.tokenize(action_vec)  # (7,)
        input_ids  = [prompt_ids ... action_ids ... eos]
        labels     = [-100       ... action_ids ... eos]
    Right-padded to the longest sequence in the batch.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    images = []
    input_ids_list = []
    labels_list = []

    for item in batch:
        images.append(_decode_image(item["image"]))

        prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids

        act_vec = np.asarray(item["action"], dtype=np.float32)
        act_ids = action_tokenizer.tokenize(act_vec).tolist()

        ids = prompt_ids + act_ids + [eos_id]
        lbl = [-100] * len(prompt_ids) + act_ids + [eos_id]

        input_ids_list.append(ids)
        labels_list.append(lbl)

    max_len = max(len(x) for x in input_ids_list)
    B = len(input_ids_list)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(lbl, dtype=torch.long)

    return {
        "image": torch.stack(images, dim=0),  # (B, 3, 384, 384)
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    llm = "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(llm)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    at = ActionTokenizer(tok, n_bins=256)

    train_loader = get_dataloader(tok, at, batch_size=4, split="train")
    print("Train Loader")
    for i, batch in enumerate(train_loader):
        print(f"--- Batch {i} ---")
        for k, v in batch.items():
            print(f"{k}: {tuple(v.shape)} ({v.dtype})")
        break

    print("\nValidation Loader")
    val_loader = get_dataloader(tok, at, batch_size=4, split="val")
    for i, batch in enumerate(val_loader):
        print(f"--- Batch {i} ---")
        for k, v in batch.items():
            print(f"{k}: {tuple(v.shape)} ({v.dtype})")
        break
