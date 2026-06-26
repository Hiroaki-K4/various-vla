"""Diagnose marginal-action collapse: does the model condition on the image?

Loads the eval checkpoint and runs model.generate() on several visibly
different *training* frames. Prints, per frame:
  - predicted action token ids + decoded action
  - ground-truth action token ids + action (teacher-forced target)

Interpretation:
  * Predicted tokens IDENTICAL across different frames  -> model ignores vision
    (collapse). Fix training (LR / projector warmup / epochs), not eval.
  * Predicted tokens DIFFER and ~match GT                -> model is fine on
    train data; a constant *rollout* then points at an eval/env issue.
  * Predicted tokens DIFFER but don't match GT           -> undertrained.

Usage:
    python diagnose_collapse.py \
        --model_path checkpoints/libero_spatial \
        --dataset_dir libero/libero/datasets/libero_spatial_384 \
        --n 6
"""

import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

from action_normalizer import ActionNormalizer
from action_tokenizer import ActionTokenizer
from eval_libero import load_model
from libero_dataloader import LiberoDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="checkpoints/libero_spatial")
    p.add_argument("--llm_model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--dataset_dir", default="libero/libero/datasets/libero_spatial_384")
    p.add_argument("--camera", default="agentview_rgb")
    p.add_argument("--n", type=int, default=6, help="number of frames to probe")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    stats_path = f"{args.model_path}_action_stats.json"
    normalizer = ActionNormalizer.load(stats_path)

    model = load_model(args.model_path, args.llm_model_name, device)

    ds = LiberoDataset(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        split="train",
        camera=args.camera,
        action_normalizer=normalizer,
    )

    # Spread the probe frames across the dataset so they look different.
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int)
    np.set_printoptions(precision=3, suppress=True)

    preds = []
    for k in idxs:
        sample = ds[int(k)]
        image = sample["image"].unsqueeze(0).to(device)  # (1,3,384,384)
        # Reconstruct the prompt-only ids (drop the 7 appended action tokens).
        full_ids = sample["input_ids"]
        prompt_len = full_ids.shape[0] - 7
        input_ids = full_ids[:prompt_len].unsqueeze(0).to(device)
        attn = torch.ones_like(input_ids)

        gt_ids = full_ids[prompt_len:].numpy()
        gt_action = normalizer.denormalize(action_tokenizer.detokenize(gt_ids))

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model.generate(image, input_ids, attn, max_new_tokens=7)
        pred_ids = out.detach().cpu().numpy().reshape(-1)[-7:]
        pred_action = normalizer.denormalize(action_tokenizer.detokenize(pred_ids))
        preds.append(tuple(pred_ids.tolist()))

        print(
            f"\n--- sample {int(k)}  (img_mean={float(sample['image'].mean()):.3f}) ---"
        )
        print(f"  pred ids : {pred_ids}   action: {pred_action}")
        print(f"  gt   ids : {gt_ids}   action: {gt_action}")

    n_unique = len(set(preds))
    print(f"\n{n_unique}/{len(preds)} distinct predicted token sequences.")
    if n_unique == 1:
        print(
            "=> COLLAPSE: identical prediction for every image. Model ignores "
            "vision. Fix TRAINING (raise LR / projector warmup / more epochs)."
        )
    else:
        print(
            "=> Model output varies with the image; collapse is at eval/rollout "
            "time, not in the trained policy. Inspect env obs vs stored obs."
        )


if __name__ == "__main__":
    main()
