import os
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from action_normalizer import compute_action_stats
from action_tokenizer import ActionTokenizer
from libero_dataloader import get_libero_dataloader
from model_plamo import PlamoVLAModel


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate with task-level loss reporting (per-sample computation)."""
    model.eval()
    total_loss = 0
    task_losses = {}
    task_counts = {}
    batch_count = 0

    for batch_data in val_loader:
        batch_count += 1
        images = batch_data["image"].to(device)
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["labels"].to(device)
        task_names = batch_data["task_names"]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss
            per_sample_loss = outputs.per_sample_loss

        if loss is not None:
            batch_loss = loss.item()
            total_loss += batch_loss

            # Accumulate per-task losses using per-sample losses
            if isinstance(task_names, (list, tuple)) and per_sample_loss is not None:
                per_sample_loss_cpu = per_sample_loss.detach().cpu()
                for idx, task_name in enumerate(task_names):
                    if task_name not in task_losses:
                        task_losses[task_name] = 0
                        task_counts[task_name] = 0
                    task_losses[task_name] += per_sample_loss_cpu[idx].item()
                    task_counts[task_name] += 1

    model.train()

    # Normalize task losses by count
    for task_name in task_losses:
        if task_counts[task_name] > 0:
            task_losses[task_name] /= task_counts[task_name]

    avg_loss = total_loss / batch_count if batch_count > 0 else float("inf")

    return avg_loss, task_losses


def train(
    dataset_dir: str | list,
    plamo_model_name: str = "pfnet/plamo-2.1-2b-vl",
    batch_size: int = 2,
    num_epochs: int = 5,
    lr_rate: float = 1e-5,
    patience: int = 5,
    eval_interval: int = 1000,
    num_workers: int = 4,
    device=None,
    save_model_path: str | None = None,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    val_demos: int = 5,
    cameras: tuple = ("agentview_rgb",),
    warmup_ratio: float = 0.05,
    seed: int = 42,
    image_aug: bool = True,
    freeze_vision: bool = True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    print(f"Random seed set to {seed}")

    print("Loading Plamo VL model...")
    model = PlamoVLAModel(plamo_model_name, device=device)

    # Optionally freeze vision encoder in Plamo
    if freeze_vision:
        # For Plamo, freeze vision_model if it exists
        if hasattr(model.model, "vision_model"):
            for p in model.model.vision_model.parameters():
                p.requires_grad = False
            print("Vision encoder frozen")

    # Apply LoRA to language model components
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    # Note: Plamo's structure may differ - adjust target_modules if needed
    model.model = get_peft_model(model.model, lora_config)
    model.model.enable_input_require_grads()
    model.model.gradient_checkpointing_enable()
    model.model.config.use_cache = False

    # Get tokenizer (use Plamo's built-in tokenizer)
    tokenizer = model.tokenizer

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    # Compute action normalization stats
    print("Computing action normalization stats (q01/q99)...")
    action_normalizer = compute_action_stats(
        dataset_dir=dataset_dir, split="train", val_demos=val_demos
    )
    print(action_normalizer)
    if save_model_path is not None:
        stats_path = f"{save_model_path}_action_stats.json"
        action_normalizer.save(stats_path)
        print(f"Saved action stats to {stats_path}")

    # Create data loaders
    train_loader = get_libero_dataloader(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        batch_size=batch_size,
        split="train",
        num_workers=num_workers,
        val_demos=val_demos,
        cameras=cameras,
        action_normalizer=action_normalizer,
        seed=seed,
        image_aug=image_aug,
    )

    val_loader = get_libero_dataloader(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        batch_size=batch_size,
        split="val",
        num_workers=num_workers,
        val_demos=val_demos,
        cameras=cameras,
        action_normalizer=action_normalizer,
        val_step_ratio=0.2,
        image_aug=False,
    )

    print(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler("cuda")

    total_steps = (num_epochs * len(train_loader)) // gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Scheduler: cosine warmup {warmup_steps} / {total_steps} steps")

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    if freeze_vision and hasattr(model.model, "vision_model"):
        model.model.vision_model.eval()

    print("Starting training...")

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images, input_ids, attention_mask, labels)
                loss = outputs.loss / gradient_accumulation_steps

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: invalid loss at step {i} (loss={loss})")
                continue

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(
                {"train_loss": f"{loss.item() * gradient_accumulation_steps:.4f}"}
            )

            if (i + 1) % eval_interval == 0 and (
                i + 1
            ) % gradient_accumulation_steps == 0:
                torch.cuda.empty_cache()
                val_loss, task_losses = evaluate(model, val_loader, device)
                print(f"\nStep {i + 1} | Val Loss: {val_loss:.4f}")
                for task_name, loss_val in task_losses.items():
                    print(f"  {task_name}: {loss_val:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_model_path is not None:
                        os.makedirs(save_model_path, exist_ok=True)
                        model.save_checkpoint(f"{save_model_path}/model.pth")
                        print("New best model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    print(f"Best Val Loss: {best_val_loss:.4f}")
                    return best_val_loss

    print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    train(
        dataset_dir=[
            "libero/libero/datasets/libero_spatial_256",
        ],
        plamo_model_name="pfnet/plamo-2.1-2b-vl",
        batch_size=2,
        num_epochs=5,
        lr_rate=1e-5,
        patience=5,
        eval_interval=1000,
        num_workers=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_model_path="checkpoints/libero_plamo",
        gradient_accumulation_steps=2,
        lora_r=32,
        lora_alpha=128,
        val_demos=5,
        cameras=("agentview_rgb",),
        image_aug=True,
        freeze_vision=True,
    )
