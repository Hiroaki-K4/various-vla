import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libero"))

import argparse
import os
import sys
import textwrap

import cv2
import imageio
import numpy as np
import torch
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
from libero_dataloader import PROMPT_TEMPLATE
from model_plamo import PlamoVLAModel

EVAL_VIEWS = (("agentview", True),)


def load_model(model_path: str, device: torch.device) -> PlamoVLAModel:
    """Load Plamo-VL model with LoRA adapter or PyTorch checkpoint."""
    print("Loading Plamo-2.1-2B-VL base model...")
    model = PlamoVLAModel(device=device)

    if os.path.exists(model_path):
        # Check if it's a LoRA adapter directory
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"Loading LoRA adapter from {model_path}...")
            model.model = PeftModel.from_pretrained(model.model, model_path)
        else:
            # Try loading as a PyTorch checkpoint
            model_pth_path = os.path.join(model_path, "model.pth")
            if os.path.exists(model_pth_path):
                print(f"Loading PyTorch checkpoint from {model_pth_path}...")
                state_dict = torch.load(model_pth_path, map_location=device)
                model.model.load_state_dict(state_dict, strict=False)
            elif os.path.isfile(model_path) and model_path.endswith(".pth"):
                print(f"Loading PyTorch checkpoint from {model_path}...")
                state_dict = torch.load(model_path, map_location=device)
                model.model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model




def annotate_frame(
    frame: np.ndarray,
    instruction: str,
    step: int,
    max_steps: int,
    success: bool,
    out_size: int = 512,
) -> np.ndarray:
    """Upscale and annotate frame."""
    frame = np.ascontiguousarray(np.rot90(frame, k=2))
    frame = cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

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
    """Save frames as video."""
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
    print(f"  saved video: {path}")


def run_episode(
    env: OffScreenRenderEnv,
    model: PlamoVLAModel,
    prompt: str,
    action_tokenizer: ActionTokenizer,
    init_state: np.ndarray,
    max_steps: int,
    device: torch.device,
    instruction: str = "",
    record: bool = False,
    debug_steps: int = 0,
    debug_every: int = 0,
    action_normalizer: ActionNormalizer | None = None,
    center_crop: bool = True,
) -> tuple:
    """Run a single episode and return (success, frames)."""
    init_state = np.asarray(init_state)
    env.reset()
    obs = env.set_init_state(init_state)

    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    frames: list = []
    success = False
    for step in range(max_steps):
        raw_image = obs[f"{EVAL_VIEWS[0][0]}_image"]

        # Rotate image 180 degrees to match training convention
        processed_image = np.rot90(raw_image, k=2)

        # Center crop if needed
        if center_crop:
            h, w = processed_image.shape[:2]
            crop_h, crop_w = int(h * 0.9), int(w * 0.9)
            start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
            processed_image = processed_image[start_h : start_h + crop_h, start_w : start_w + crop_w]

        # Use Plamo processor for consistent image processing
        processor_output = model.processor(
            text=prompt,
            images=[processed_image],
            return_tensors="pt"
        )
        input_ids = processor_output["input_ids"].to(device)
        attention_mask = processor_output["attention_mask"].to(device)
        pixel_values = processor_output["pixel_values"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, attention_mask, pixel_values, max_new_tokens=7
            )

        action = action_tokenizer.decode_model_output(output_ids, action_dim=7)
        if action.ndim == 2:
            action = action[0]

        if action_normalizer is not None:
            action = action_normalizer.denormalize(action)

        if step < debug_steps:
            ids = output_ids.detach().cpu().numpy().reshape(-1)
            img_mean = float(raw_image.mean())
            np.set_printoptions(precision=3, suppress=True)
            print(
                f"    [debug] step {step:3d}  img_mean={img_mean:6.2f}  "
                f"token_ids={ids}  action={action}"
            )
        elif debug_every > 0 and step % debug_every == 0:
            np.set_printoptions(precision=3, suppress=True)
            xyz_mag = float(np.abs(action[:3]).max())
            print(
                f"    [debug] step {step:3d}  xyz={action[:3]}  rpy={action[3:6]}  "
                f"grip={action[6]:+.3f}  |xyz|max={xyz_mag:.3f}"
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
    task_suite_name: str,
    n_episodes: int,
    max_steps: int,
    device: torch.device,
    save_video_flag: bool = False,
    video_dir: str = "eval_videos",
    video_mode: str = "all",
    debug_actions: int = 0,
    debug_every: int = 0,
    center_crop: bool = True,
) -> float:
    """Evaluate model on task suite."""
    model = load_model(model_path, device)
    tokenizer = model.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    stats_path = f"{model_path}_action_stats.json"
    if os.path.exists(stats_path):
        action_normalizer = ActionNormalizer.load(stats_path)
        print(f"Loaded action stats from {stats_path}")
    else:
        action_normalizer = None
        print(
            f"WARNING: {stats_path} not found; running without action denormalization."
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
            camera_names=[cam for cam, _ in EVAL_VIEWS],
            camera_heights=256,
            camera_widths=256,
            use_camera_obs=True,
            ignore_done=True,
        )
        env.seed(0)
        env.reset()

        init_states = task_suite.get_task_init_states(task_id)

        prompt = PROMPT_TEMPLATE.format(instruction=task.language)

        successes = 0
        for ep in range(n_episodes):
            init_state = init_states[ep % len(init_states)]
            record = save_video_flag and (video_mode != "first" or ep == 0)
            success, frames = run_episode(
                env,
                model,
                prompt,
                action_tokenizer,
                init_state,
                max_steps,
                device,
                instruction=task.language,
                record=record,
                debug_steps=debug_actions if (task_id == 0 and ep == 0) else 0,
                debug_every=debug_every if (task_id == 0 and ep == 0) else 0,
                action_normalizer=action_normalizer,
                center_crop=center_crop,
            )
            successes += int(success)
            status = "SUCCESS" if success else "FAIL"
            print(
                f"  Task {task_id:2d} ep {ep + 1:2d}/{n_episodes}  {status}"
            )

            if record:
                video_name = f"{task_suite_name}_task{task_id:02d}_ep{ep:02d}_{status}.mp4"
                if video_mode == "fail" and success:
                    pass  # Don't save successful episodes in "fail" mode
                else:
                    save_video(frames, f"{video_dir}/{video_name}")

        task_success_rate = successes / n_episodes
        overall_successes += successes
        overall_total += n_episodes
        print(f"  Task {task_id:2d} success rate: {task_success_rate:.1%}\n")

    overall_success_rate = overall_successes / overall_total
    print(f"\n{'='*50}")
    print(f"Overall success rate: {overall_success_rate:.1%}")
    print(f"{'='*50}\n")

    return overall_success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Plamo-VL VLA on LIBERO")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints_plamo_vl_2b/libero_plamo",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_spatial",
        help="Task suite name (libero_spatial, libero_object, libero_goal)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10, help="Episodes per task"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Max steps per episode"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save episode videos"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="eval_videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        default="all",
        choices=["first", "fail", "all"],
        help="Which videos to save",
    )
    parser.add_argument(
        "--debug-actions",
        type=int,
        default=0,
        help="Debug first N action steps",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=0,
        help="Debug every N steps",
    )
    parser.add_argument(
        "--no_center_crop",
        action="store_false",
        dest="center_crop",
        help="Disable center crop at 90% scale (default is enabled)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        model_path=args.model_path,
        task_suite_name=args.task_suite,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        device=device,
        save_video_flag=args.save_video,
        video_dir=args.video_dir,
        video_mode=args.video_mode,
        debug_actions=args.debug_actions,
        debug_every=args.debug_every,
        center_crop=args.center_crop,
    )
