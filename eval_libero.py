import argparse
import os
import sys
import textwrap

import cv2
import imageio
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

from action_normalizer import ActionNormalizer
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
    """(H, W, 3) uint8 → (1, 3, 384, 384) float32 in [0, 1]

    The agentview image is rendered upside down; rotate 180 degrees to match the
    orientation the model was trained on (see libero_dataloader.rotate_180).
    """
    obs_image = np.rot90(obs_image, k=2)
    image = (
        torch.from_numpy(np.ascontiguousarray(obs_image)).permute(2, 0, 1).float()
        / 255.0
    )
    image = F.interpolate(
        image.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False
    )
    return image.to(device)


def annotate_frame(
    frame: np.ndarray,
    instruction: str,
    step: int,
    max_steps: int,
    success: bool,
    out_size: int = 512,
) -> np.ndarray:
    """Upscale a raw obs frame and draw the task goal + status banner on it.

    The agentview image is rendered upside-down; rotate 180 degrees so the video
    shows the same orientation the model is fed (see preprocess_obs).
    """
    # Rotate 180 deg (matches preprocess_obs / training) and upscale for readability.
    frame = np.ascontiguousarray(np.rot90(frame, k=2))
    frame = cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # Bottom banner holding the (possibly wrapped) instruction text.
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

    # Top-left overlay: step counter and success status.
    status = "SUCCESS" if success else "running"
    color = (0, 255, 0) if success else (0, 200, 255)
    cv2.putText(
        frame,
        f"step {step}/{max_steps}  {status}",
        (8, 22),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return np.concatenate([frame, banner], axis=0)


def save_video(frames: list, path: str, fps: int = 20) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
    print(f"  saved video: {path}")


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
    instruction: str = "",
    record: bool = False,
    debug_steps: int = 0,
    action_normalizer: ActionNormalizer | None = None,
) -> tuple:
    """Returns (success, frames). frames is an empty list when record=False.

    debug_steps > 0: print the raw generated token ids and decoded action for
    the first N steps. If the action is ~constant across visibly different
    frames, the policy has collapsed to the dataset's marginal action.
    """
    init_state = np.asarray(init_state)
    env.reset()
    obs = env.set_init_state(init_state)

    # Warm up physics (same as LIBERO's official eval)
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    frames: list = []
    success = False
    for step in range(max_steps):
        raw_image = obs[f"{camera}_image"]
        image = preprocess_obs(raw_image, device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            output_ids = model.generate(
                image, input_ids, attention_mask, max_new_tokens=7
            )

        action = action_tokenizer.decode_model_output(output_ids, action_dim=7)
        if action.ndim == 2:
            action = action[0]  # (7,)

        # Map normalized [-1, 1] prediction back to raw action space the env
        # expects (must match the q01/q99 normalization used during training).
        if action_normalizer is not None:
            action = action_normalizer.denormalize(action)

        if step < debug_steps:
            ids = output_ids.detach().cpu().numpy().reshape(-1)
            img_mean = float(raw_image.mean())  # cheap proxy that the input differs
            np.set_printoptions(precision=3, suppress=True)
            print(
                f"    [debug] step {step:3d}  img_mean={img_mean:6.2f}  "
                f"token_ids={ids}  action={action}"
            )

        obs, _, done, _ = env.step(action)
        success = env.check_success()

        if record:
            frames.append(
                annotate_frame(raw_image, instruction, step + 1, max_steps, success)
            )

        if success or done:
            break

    return success, frames


def evaluate(
    model_path: str,
    llm_model_name: str,
    task_suite_name: str,
    n_episodes: int,
    max_steps: int,
    device: torch.device,
    camera: str = "agentview",
    save_video_flag: bool = False,
    video_dir: str = "eval_videos",
    video_mode: str = "all",
    debug_actions: int = 0,
) -> float:
    model = load_model(model_path, llm_model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    # Load the q01/q99 action stats saved during training so predictions are
    # denormalized back to the env's raw action space.
    stats_path = f"{model_path}_action_stats.json"
    if os.path.exists(stats_path):
        action_normalizer = ActionNormalizer.load(stats_path)
        print(f"Loaded action stats from {stats_path}")
    else:
        action_normalizer = None
        print(
            f"WARNING: {stats_path} not found; running without action "
            "denormalization (predictions assumed already in raw space)."
        )

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
            camera_heights=384,
            camera_widths=384,
            use_camera_obs=True,
            ignore_done=True,
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
            # "first": only the first episode; "fail"/"all": record then decide.
            record = save_video_flag and (video_mode != "first" or ep == 0)
            success, frames = run_episode(
                env,
                model,
                input_ids,
                attention_mask,
                action_tokenizer,
                init_state,
                max_steps,
                device,
                camera,
                instruction=task.language,
                record=record,
                debug_steps=debug_actions if (task_id == 0 and ep == 0) else 0,
                action_normalizer=action_normalizer,
            )
            successes += int(success)
            status = "SUCCESS" if success else "FAIL"
            print(
                f"  Task {task_id:2d} ep {ep + 1:2d}/{n_episodes}  {status}"
                f"  [{task.language[:60]}]"
            )

            if record and (video_mode != "fail" or not success):
                fname = f"task{task_id:02d}_ep{ep + 1:02d}_{status}.mp4"
                save_video(frames, os.path.join(video_dir, task_suite_name, fname))

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
        default=1000,
        help="Max steps per episode (LIBERO default horizon is 1000)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera name (obs key will be {camera}_image)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save rollout videos (mp4) with the task goal overlaid",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="eval_videos",
        help="Directory to write videos into (subfoldered per task suite)",
    )
    parser.add_argument(
        "--video_mode",
        type=str,
        default="all",
        choices=["all", "fail", "first"],
        help="Which episodes to save: all / only failures / only first episode per task",
    )
    parser.add_argument(
        "--debug_actions",
        type=int,
        default=0,
        help="Print token ids + decoded action for the first N steps of the first "
        "episode (diagnose policy collapse: action ~constant across frames)",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        llm_model_name=args.llm_model_name,
        task_suite_name=args.task_suite,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        device=torch.device(args.device),
        camera=args.camera,
        save_video_flag=args.save_video,
        video_dir=args.video_dir,
        video_mode=args.video_mode,
        debug_actions=args.debug_actions,
    )
