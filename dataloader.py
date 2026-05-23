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


def get_dataloader(batch_size=8, split="train", seq_len=10, traj_stride=1):
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
            traj_instructions = []
            traj_images = []
            traj_actions = []

            episodes = examples.get("data.pickle", examples.get("steps", []))
            for episode in episodes:
                steps = episode["steps"]

                valid_steps = []
                for step in steps:
                    obs = step.get("observation", {})
                    image_field = obs.get("image", None)
                    image_bytes = None
                    if isinstance(image_field, dict):
                        image_bytes = image_field.get("bytes", None)
                    elif isinstance(image_field, (bytes, bytearray)):
                        image_bytes = bytes(image_field)

                    if image_bytes is not None:
                        instruction = obs.get("natural_language_instruction", "")
                        if isinstance(instruction, bytes):
                            instruction = instruction.decode("utf-8", errors="ignore")

                        valid_steps.append(
                            {
                                "instruction": instruction,
                                "image": image_bytes,
                                "action": step["action"],
                            }
                        )

                if len(valid_steps) < seq_len:
                    continue
                else:
                    for start_idx in range(
                        0, len(valid_steps) - seq_len + 1, traj_stride
                    ):
                        end_idx = start_idx + seq_len
                        sub_steps = valid_steps[start_idx:end_idx]

                        traj_instructions.append([s["instruction"] for s in sub_steps])
                        traj_images.append([s["image"] for s in sub_steps])
                        traj_actions.append([s["action"] for s in sub_steps])

            return {
                "instruction": traj_instructions,
                "image": traj_images,
                "action": traj_actions,
            }

        ds = ds.map(chunk_episodes, batched=True, remove_columns=ds.column_names)

        ds_list.append(ds)

    combined_ds = datasets.interleave_datasets(ds_list, seed=42)

    return DataLoader(combined_ds, batch_size=batch_size, collate_fn=collate_fn)


def _decode_image(image_bytes, image_size):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(image_size, Image.BILINEAR)
    return torch.from_numpy(np.array(img))


def collate_fn(batch):
    out_instructions = []
    out_images = []
    out_actions = []
    for item in batch:
        out_instructions.append(item["instruction"])

        traj_imgs = torch.stack(
            [_decode_image(img_b, (256, 256)) for img_b in item["image"]]
        )
        out_images.append(traj_imgs)

        traj_acts = default_collate(item["action"])
        out_actions.append(traj_acts)

    final_images = torch.stack(out_images)

    return {
        "instruction": out_instructions,
        "image": final_images,
        "action": out_actions,
    }


if __name__ == "__main__":
    loader = get_dataloader(batch_size=4, split="train", seq_len=10)
    for i, batch in enumerate(loader):
        print(f"--- Batch {i} ---")
        print(
            "Instruction length: ",
            len(batch["instruction"]),
            "x",
            len(batch["instruction"][0]),
        )
        print("Image tensor shape: ", batch["image"].shape)
        print(
            "Action (per-sample dict keys): ", [list(a.keys()) for a in batch["action"]]
        )
        break
