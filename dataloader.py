import io

import datasets
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

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


def get_dataloader(batch_size=8, split="train"):
    ds_list = []
    for name in DATASETS:
        ds = datasets.load_dataset(
            "jxu124/OpenX-Embodiment",
            name,
            streaming=True,
            split=split,
            trust_remote_code=True,
        )

        def flatten_steps(examples):
            all_intructions = []
            all_images = []
            all_actions = []

            episodes = examples.get("data.pickle", examples.get("steps", []))
            for episode in episodes:
                for step in episode["steps"]:
                    obs = step.get("observation", {})

                    instruction = obs.get("natural_language_instruction", "")
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode("utf-8", errors="ignore")

                    image_field = obs.get("image", None)
                    image_bytes = None
                    if isinstance(image_field, dict):
                        image_bytes = image_field.get("bytes", None)
                    elif isinstance(image_field, (bytes, bytearray)):
                        image_bytes = bytes(image_field)

                    if image_bytes is None:
                        # Skip steps without an image so collate doesn't see None.
                        continue

                    all_intructions.append(instruction)
                    all_images.append(image_bytes)
                    all_actions.append(step["action"])

            return {
                "instruction": all_intructions,
                "image": all_images,
                "action": all_actions,
            }

        ds = ds.map(flatten_steps, batched=True, remove_columns=ds.column_names)

        ds_list.append(ds)

    combined_ds = datasets.interleave_datasets(ds_list, seed=42)

    return DataLoader(combined_ds, batch_size=batch_size, collate_fn=collate_fn)


def _decode_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return torch.from_numpy(np.array(img))


def collate_fn(batch):
    instructions = [item["instruction"] for item in batch]
    images = [_decode_image(item["image"]) for item in batch]
    actions = default_collate([item["action"] for item in batch])
    return {"instruction": instructions, "image": images, "action": actions}


if __name__ == "__main__":
    loader = get_dataloader(batch_size=8, split="train")
    for i, batch in enumerate(loader):
        print("Batch data")
        print(batch)
        input()
