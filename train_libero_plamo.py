import os
import random
from functools import partial

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

        # input_ids = prompt_ids + action_token_ids (7 tokens)
        # labels = [-100, -100, ..., -100, action_token_0, ..., action_token_6]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Extract action tokens (last 7 tokens where labels != -100)
        action_dim = 7
        prompt_ids = input_ids[:-action_dim]

        # Decode only prompt part (without action tokens)
        # This avoids polluting the text with action token representations
        text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        text_list.append(text)

        # Store action token ids for label reconstruction
        action_token_ids_list.append(labels[-action_dim:])
        task_names.append(sample["task_name"])

    # Stack images (take first camera view from each sample)
    batch_images = torch.stack(
        [img[0] if img.dim() == 4 else img for img in images_list]
    )  # (B, 3, H, W)

    # Process each sample individually with Plamo processor (expects single text + image)
    # Key: processor only receives prompt text, not action tokens
    # We'll append action tokens to input_ids after processing for proper teacher forcing
    batch_processed = []
    for i in range(len(batch)):
        pil_image = _tensor_to_pil(batch_images[i])
        text = text_list[i]
        processed_sample = processor(
            text=text,
            images=pil_image,
            return_tensors="pt",
        )
        batch_processed.append(processed_sample)

    # Append action tokens to input_ids for proper teacher forcing
    # Structure: [image_tokens] + [prompt_tokens] + [action_tokens]
    for i, p in enumerate(batch_processed):
        action_tokens = action_token_ids_list[i]  # (7,)
        input_ids = p["input_ids"].squeeze(0)  # (prompt_len,)
        attention_mask = p["attention_mask"].squeeze(0)  # (prompt_len,)

        # Concatenate action tokens
        input_ids_with_action = torch.cat([input_ids, action_tokens])
        attention_mask_with_action = torch.cat(
            [
                attention_mask,
                torch.ones(action_tokens.numel(), dtype=attention_mask.dtype),
            ]
        )

        p["input_ids"] = input_ids_with_action.unsqueeze(0)
        p["attention_mask"] = attention_mask_with_action.unsqueeze(0)

    # Extract padded sequences
    input_ids_list = [p["input_ids"].squeeze(0) for p in batch_processed]
    attention_mask_list = [p["attention_mask"].squeeze(0) for p in batch_processed]

    # Pad input_ids and attention_mask to max length
    max_seq_len = max(ids.shape[0] for ids in input_ids_list)
    input_ids_padded = torch.full(
        (len(batch), max_seq_len), processor.tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask_padded = torch.zeros((len(batch), max_seq_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        input_ids_padded[i, : ids.shape[0]] = ids
        attention_mask_padded[i, : mask.shape[0]] = mask

    # Collect pixel_values (concatenated across batch)
    pixel_values_list = [p["pixel_values"] for p in batch_processed]
    pixel_values = torch.cat(pixel_values_list, dim=0)

    processed = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "pixel_values": pixel_values,
    }

    # Create labels: supervise action tokens, mask everything else (image + prompt)
    # input_ids structure: [image_tokens] + [prompt_tokens] + [action_tokens]
    # labels structure:    [-100        ] + [-100           ] + [action_tokens]
    batch_size = processed["input_ids"].shape[0]
    seq_len = processed["input_ids"].shape[1]

    labels_batch = torch.full(
        (batch_size, seq_len), fill_value=ignore_index, dtype=torch.long
    )

    # Supervise only action tokens at the end of sequence
    # Since we appended action tokens after processor output, we know:
    # - last action_dim tokens are the action tokens
    action_dim = 7
    for b in range(batch_size):
        action_tokens = action_token_ids_list[b]  # (7,)

        # Find last real token position
        attention_mask = processed["attention_mask"][b]
        last_real_pos = int(attention_mask.sum().item())

        # Action tokens are at the end, right before padding
        start = last_real_pos - action_dim
        end = last_real_pos

        # Place action token labels only at the action positions
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

    # Apply LoRA to language model components (required)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    # Apply LoRA adapter
    model.model = get_peft_model(model.model, lora_config)
    model.model.config.use_cache = False
    print(
        f"LoRA adapter applied: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}"
    )
    model.model.print_trainable_parameters()

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

    # Save action stats for later use in evaluation
    if save_model_path is not None:
        # Create directory early to save stats
        os.makedirs(save_model_path, exist_ok=True)
        stats_path = os.path.join(save_model_path, "action_stats.json")
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
                        # Save only LoRA adapter (strict format)
                        model.save_checkpoint(save_model_path)
                        # Verify LoRA files exist (support both .bin and .safetensors)
                        adapter_config = os.path.join(
                            save_model_path, "adapter_config.json"
                        )
                        adapter_model_bin = os.path.join(
                            save_model_path, "adapter_model.bin"
                        )
                        adapter_model_safetensors = os.path.join(
                            save_model_path, "adapter_model.safetensors"
                        )
                        has_config = os.path.exists(adapter_config)
                        has_model = os.path.exists(adapter_model_bin) or os.path.exists(
                            adapter_model_safetensors
                        )
                        if has_config and has_model:
                            print("✓ New best LoRA adapter saved!")
                        else:
                            print("WARNING: LoRA adapter files not properly saved!")
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
            "../../various-vla/libero/libero/datasets/libero_spatial_384",
        ],
        plamo_model_name="pfnet/plamo-2.1-2b-vl",
        batch_size=1,
        num_epochs=5,
        lr_rate=1e-5,
        patience=5,
        eval_interval=1000,
        num_workers=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_model_path="checkpoints_plamo_vl_2b_384_hand/libero_plamo_vl_2b",
        gradient_accumulation_steps=4,
        lora_r=32,
        lora_alpha=128,
        val_demos=5,
        cameras=("agentview_rgb", "eye_in_hand_rgb"),
        image_aug=False,
        freeze_vision=True,
    )
