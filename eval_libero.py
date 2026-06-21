import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer

# Fix libero namespace collision (libero/ submodule vs installed package).
for _i, _finder in enumerate(sys.meta_path):
    if getattr(_finder, "__name__", "") == "_EditableFinder" and "libero" in getattr(
        _finder, "__module__", ""
    ):
        sys.meta_path.insert(0, sys.meta_path.pop(_i))
        break

from action_tokenizer import ActionTokenizer
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero_dataloader import IMAGE_SIZE, PROMPT_TEMPLATE
from model import VLAModel


def load_model(model_path: str, llm_model_name: str, device: torch.device) -> VLAModel:
    print("Loading base model...")
    model = VLAModel(llm_model_name, device=device)

    print(f"Loading LoRA adapter from {model_path} ...")
    model.llm = PeftModel.from_pretrained(model.llm, model_path)

    projector_path = f"{model_path}_projector.pth"
    dino_path = f"{model_path}_dino.pth"
    siglip_path = f"{model_path}_siglip.pth"

    if os.path.exists(projector_path):
        print(f"Loading projector from {projector_path}")
        model.projector.load_state_dict(torch.load(projector_path, map_location=device))

    if os.path.exists(dino_path):
        print(f"Loading DINO from {dino_path}")
        model.dino.load_state_dict(torch.load(dino_path, map_location=device))

    if os.path.exists(siglip_path):
        print(f"Loading SigLIP from {siglip_path}")
        model.siglip.load_state_dict(torch.load(siglip_path, map_location=device))

    model.eval()
    return model


def preprocess_obs(obs_image: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, 384, 384) float32 in [0, 1]"""
    image = torch.from_numpy(obs_image).permute(2, 0, 1).float() / 255.0
    image = F.interpolate(
        image.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False
    )
    return image.to(device)


def run_episode(
    env: OffScreenRenderEnv,
    model: VLAModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_tokenizer: ActionTokenizer,
    init_state: np.ndarray,
    max_steps: int,
    device: torch.device,
    camera: str,
) -> bool:
    init_state = np.asarray(init_state)
    env.reset()
    obs = env.set_init_state(init_state)

    # Warm up physics (same as LIBERO's official eval)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    for _ in range(max_steps):
        image = preprocess_obs(obs[f"{camera}_image"], device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            output_ids = model.generate(
                image, input_ids, attention_mask, max_new_tokens=7
            )

        action = action_tokenizer.decode_model_output(output_ids, action_dim=7)
        if action.ndim == 2:
            action = action[0]  # (7,)

        obs, _, done, _ = env.step(action)

        if env.check_success():
            return True
        if done:
            break

    return env.check_success()


def evaluate(
    model_path: str,
    llm_model_name: str,
    task_suite_name: str,
    n_episodes: int,
    max_steps: int,
    device: torch.device,
    camera: str = "agentview",
) -> float:
    model = load_model(model_path, llm_model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    n_tasks = task_suite.n_tasks

    print(
        f"\nEvaluating on {task_suite_name} "
        f"({n_tasks} tasks, {n_episodes} episodes each)\n"
    )

    overall_successes = 0
    overall_total = 0

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )

        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_names=[camera],
            camera_heights=128,
            camera_widths=128,
            use_camera_obs=True,
        )
        env.seed(0)
        env.reset()

        init_states = task_suite.get_task_init_states(task_id)

        # Build prompt tokens once per task (instruction is task-level)
        prompt = PROMPT_TEMPLATE.format(instruction=task.language)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        successes = 0
        for ep in range(n_episodes):
            init_state = init_states[ep % len(init_states)]
            success = run_episode(
                env,
                model,
                input_ids,
                attention_mask,
                action_tokenizer,
                init_state,
                max_steps,
                device,
                camera,
            )
            successes += int(success)
            status = "SUCCESS" if success else "FAIL"
            print(
                f"  Task {task_id:2d} ep {ep + 1:2d}/{n_episodes}  {status}"
                f"  [{task.language[:60]}]"
            )

        rate = successes / n_episodes
        print(
            f"  -> Task {task_id:2d} success: {successes}/{n_episodes} ({rate:.1%})\n"
        )
        overall_successes += successes
        overall_total += n_episodes

        env.close()

    overall_rate = overall_successes / overall_total
    print(f"Overall: {overall_successes}/{overall_total} ({overall_rate:.1%})")
    return overall_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VLA model in LIBERO simulation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/libero_spatial",
        help="Path to saved LoRA checkpoint directory (also expects _projector.pth, _dino.pth, _siglip.pth siblings)",
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_100"],
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Max steps per episode (LIBERO default horizon is 1000)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera name (obs key will be {camera}_image)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        llm_model_name=args.llm_model_name,
        task_suite_name=args.task_suite,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        device=torch.device(args.device),
        camera=args.camera,
    )
