import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from action_normalizer import compute_action_stats
from action_tokenizer import ActionTokenizer
from libero_dataloader import LiberoDataset, get_libero_dataloader
from model_plamo import PlamoVLAModel


def _tensor_to_pil(image_tensor):
    """Convert (3, H, W) float32 [0,1] tensor to PIL Image"""
    from PIL import Image

    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def _pad_1d_tensors(sequences, pad_value: int):
    """Pad 1D tensors to same length for batching."""
    max_len = max(seq.numel() for seq in sequences)
    output = torch.full(
        (len(sequences), max_len),
        fill_value=pad_value,
        dtype=sequences[0].dtype,
    )
    for i, seq in enumerate(sequences):
        output[i, : seq.numel()] = seq
    return output


def _collate_fn_plamo(batch, processor, tokenizer, ignore_index: int = -100):
    """
    Collate function for Plamo VLA training.

    Processor is applied to images + text to get proper input_ids with image_token inserted.
    Labels are reconstructed based on processor output to align with modified input_ids.
    """
    images_list = []
    text_list = []
    action_token_ids_list = []
    task_names = []

    for sample in batch:
        images_list.append(sample["image"])  # (V, 3, H, W)

        # Text is prompt + action tokens (already in token form)
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decode input_ids to get text
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        text_list.append(text)
        action_token_ids_list.append(labels)
        task_names.append(sample["task_name"])

    # Stack images (take first camera view from each sample)
    batch_images = torch.stack(
        [img[0] if img.dim() == 4 else img for img in images_list]
    )  # (B, 3, H, W)

    # Convert to PIL for processor
    pil_images = [_tensor_to_pil(batch_images[i]) for i in range(batch_images.shape[0])]

    # Process with Plamo processor
    processed = processor(
        text=text_list,
        images=pil_images,
        padding=True,
        return_tensors="pt",
    )

    # Get prompt length before action tokens (from original input_ids)
    # After processor, input_ids has: [image_tokens] + [prompt] + [action_tokens]
    # We need to mask prompt but not image_tokens, and keep action tokens

    # Create labels: only supervise action tokens
    batch_size = processed["input_ids"].shape[0]
    seq_len = processed["input_ids"].shape[1]

    labels_batch = torch.full(
        (batch_size, seq_len), fill_value=ignore_index, dtype=torch.long
    )

    # Use actual action_token_ids collected from dataloader
    # Each sample has action_token_ids which we collected earlier
    for b in range(batch_size):
        action_tokens = action_token_ids_list[b]
        action_dim = action_tokens.numel()

        # Find actual sequence length from attention_mask (last 1 position)
        attention_mask = processed["attention_mask"][b]
        last_real_pos = attention_mask.sum().item()
        end = int(last_real_pos)
        start = end - action_dim

        # Supervise action tokens at their actual position
        if start >= 0:
            labels_batch[b, start:end] = action_tokens

    return {
        "image": batch_images,
        "input_ids": processed["input_ids"],
        "attention_mask": processed["attention_mask"],
        "pixel_values": processed["pixel_values"],
        "labels": labels_batch,
        "task_names": task_names,
    }


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
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        pixel_values = batch_data["pixel_values"].to(device)
        labels = batch_data["labels"].to(device)
        task_names = batch_data["task_names"]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss

        if loss is not None:
            batch_loss = loss.item()
            total_loss += batch_loss

            # Compute per-sample loss for accurate task-level metrics
            if isinstance(task_names, (list, tuple)) and hasattr(outputs, "logits"):
                logits = outputs.logits
                per_sample_loss_list = []
                for b in range(logits.shape[0]):
                    # Causal LM: shift logits and labels (predict next token)
                    sample_logits = logits[b, :-1].unsqueeze(0)  # (1, T-1, vocab_size)
                    sample_labels = labels[b, 1:].unsqueeze(0)  # (1, T-1)
                    sample_loss = F.cross_entropy(
                        sample_logits.view(-1, sample_logits.shape[-1]),
                        sample_labels.view(-1),
                        reduction="mean",
                        ignore_index=-100,
                    )
                    per_sample_loss_list.append(sample_loss.item())

                # Accumulate per-task losses using per-sample losses
                for idx, task_name in enumerate(task_names):
                    if task_name not in task_losses:
                        task_losses[task_name] = 0
                        task_counts[task_name] = 0
                    task_losses[task_name] += per_sample_loss_list[idx]
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

    # Create data loaders with Plamo-specific collate function
    from torch.utils.data import DataLoader

    from libero_dataloader import LiberoDataset

    train_dataset = LiberoDataset(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        split="train",
        val_demos=val_demos,
        cameras=cameras,
        action_normalizer=action_normalizer,
        image_aug=image_aug,
    )

    val_dataset = LiberoDataset(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        split="val",
        val_demos=val_demos,
        cameras=cameras,
        action_normalizer=action_normalizer,
        val_step_ratio=0.2,
        image_aug=False,
    )

    # Use Plamo-specific collate function
    collate_fn = partial(
        _collate_fn_plamo, processor=model.processor, tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
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
            "../../various-vla/libero/libero/datasets/libero_spatial_256",
        ],
        plamo_model_name="pfnet/plamo-2.1-2b-vl",
        batch_size=2,
        num_epochs=5,
        lr_rate=1e-5,
        patience=5,
        eval_interval=1000,
        num_workers=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_model_path="checkpoints/libero_plamo_vl_2b",
        gradient_accumulation_steps=2,
        lora_r=32,
        lora_alpha=128,
        val_demos=5,
        cameras=("agentview_rgb",),
        image_aug=True,
        freeze_vision=True,
    )
