import io

import datasets
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from action_tokenizer import ActionTokenizer

IMAGE_SIZE = (384, 384)
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


DATASETS = [
    "fractal20220817_data",
    "kuka",
    # 'bridge',
    # 'taco_play',
    # 'jaco_play',
    # 'berkeley_cable_routing',
    # 'roboturk',
    # 'nyu_door_opening_surprising_effectiveness',
    # 'viola',
    # 'berkeley_autolab_ur5',
    # 'toto',
    # 'language_table',
    # 'columbia_cairlab_pusht_real',
    # 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
    # 'nyu_rot_dataset_converted_externally_to_rlds',
    # 'stanford_hydra_dataset_converted_externally_to_rlds',
    # 'austin_buds_dataset_converted_externally_to_rlds',
    # 'nyu_franka_play_dataset_converted_externally_to_rlds',
    # 'maniskill_dataset_converted_externally_to_rlds',
    # 'cmu_franka_exploration_dataset_converted_externally_to_rlds',
    # 'ucsd_kitchen_dataset_converted_externally_to_rlds',
    # 'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    # 'austin_sailor_dataset_converted_externally_to_rlds',
    # 'austin_sirius_dataset_converted_externally_to_rlds',
    # 'bc_z',
    # 'usc_cloth_sim_converted_externally_to_rlds',
    # 'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    # 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
    # 'utokyo_saytap_converted_externally_to_rlds',
    # 'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
    # 'utokyo_xarm_bimanual_converted_externally_to_rlds',
    # 'robo_net',
    # 'berkeley_mvp_converted_externally_to_rlds',
    # 'berkeley_rpt_converted_externally_to_rlds',
    # 'kaist_nonprehensile_converted_externally_to_rlds',
    # 'stanford_mask_vit_converted_externally_to_rlds',
    # 'tokyo_u_lsmo_converted_externally_to_rlds',
    # 'dlr_sara_pour_converted_externally_to_rlds',
    # 'dlr_sara_grid_clamp_converted_externally_to_rlds',
    # 'dlr_edan_shared_control_converted_externally_to_rlds',
    # 'asu_table_top_converted_externally_to_rlds',
    # 'stanford_robocook_converted_externally_to_rlds',
    # 'eth_agent_affordances',
    # 'imperialcollege_sawyer_wrist_cam',
    # 'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
    # 'uiuc_d3field',
    # 'utaustin_mutex',
    # 'berkeley_fanuc_manipulation',
    # 'cmu_play_fusion',
    # 'cmu_stretch',
    # 'berkeley_gnm_recon',
    # 'berkeley_gnm_cory_hall',
    # 'berkeley_gnm_sac_son'
]


IMAGE_SIZE = (384, 384)


def get_dataloader(
    tokenizer,
    action_tokenizer: ActionTokenizer,
    batch_size: int = 8,
    split: str = "train",
    num_workers: int = 0,
):
    ds_list = []
    for name in DATASETS:
        ds = datasets.load_dataset(
            "jxu124/OpenX-Embodiment",
            name,
            streaming=True,
            split=split,
            trust_remote_code=True,
        )

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

                    out_instr.append(instruction)
                    out_img.append(image_bytes)
                    out_act.append(step["action"])

            return {
                "instruction": out_instr,
                "image": out_img,
                "action": out_act,
            }

        ds = ds.map(chunk_episodes, batched=True, remove_columns=ds.column_names)
        ds_list.append(ds)

    combined_ds = datasets.interleave_datasets(ds_list, seed=42)

    def _collate(batch):
        return collate_fn(batch, tokenizer, action_tokenizer)

    return DataLoader(
        combined_ds,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
    )


def _decode_image(image_bytes, image_size=IMAGE_SIZE):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(image_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def _action_dict_to_vec(action) -> np.ndarray:
    """
    OpenX action dict -> 7-dim vector [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    wv = np.asarray(action["world_vector"], dtype=np.float32).reshape(-1)
    rot = np.asarray(action["rotation_delta"], dtype=np.float32).reshape(-1)
    grip = np.asarray(action["gripper_closedness_action"], dtype=np.float32).reshape(-1)
    return np.concatenate([wv, rot, grip], axis=0)  # (7,)


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

        act_vec = _action_dict_to_vec(item["action"])
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

    loader = get_dataloader(tok, at, batch_size=4, split="train")
    for i, batch in enumerate(loader):
        print(f"--- Batch {i} ---")
        for k, v in batch.items():
            print(f"{k}: {tuple(v.shape)} ({v.dtype})")
        break

    # TODO: Split train data for validation data
