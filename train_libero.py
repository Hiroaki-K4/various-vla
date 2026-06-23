import os

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from action_tokenizer import ActionTokenizer
from libero_dataloader import get_libero_dataloader
from model import VLAModel
from train import evaluate


def train(
    dataset_dir: str,
    llm_model_name: str,
    batch_size: int,
    num_epochs: int,
    lr_rate: float,
    patience: int,
    eval_interval: int,
    num_workers: int,
    device: torch.device,
    save_model_path: str | None,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    val_demos: int = 5,
    camera: str = "agentview_rgb",
    warmup_ratio: float = 0.05,
):
    print("Loading models...")
    model = VLAModel(llm_model_name, device=device)

    # Freeze vision encoders (DINOv2 + SigLIP): keep pretrained features fixed
    for p in model.dino.parameters():
        p.requires_grad = False
    for p in model.siglip.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.enable_input_require_grads()
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    scaler = torch.amp.GradScaler("cuda")
    _scheduler = None  # created after train_loader is built

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    action_tokenizer = ActionTokenizer(tokenizer, n_bins=256)

    train_loader = get_libero_dataloader(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        batch_size=batch_size,
        split="train",
        num_workers=num_workers,
        val_demos=val_demos,
        camera=camera,
    )
    val_loader = get_libero_dataloader(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        batch_size=batch_size,
        split="val",
        num_workers=num_workers,
        val_demos=val_demos,
        camera=camera,
    )

    print(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    total_steps = (num_epochs * len(train_loader)) // gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    _scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Scheduler: cosine warmup {warmup_steps} / {total_steps} steps")

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    # Frozen vision encoders stay in eval mode (no dropout / norm-stat updates)
    model.dino.eval()
    model.siglip.eval()
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
                _scheduler.step()

            pbar.set_postfix(
                {"train_loss": f"{loss.item() * gradient_accumulation_steps:.4f}"}
            )

            if (i + 1) % eval_interval == 0 and (
                i + 1
            ) % gradient_accumulation_steps == 0:
                torch.cuda.empty_cache()
                val_loss = evaluate(model, val_loader, device)
                print(f"\nStep {i + 1} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_model_path is not None:
                        os.makedirs(save_model_path, exist_ok=True)
                        model.llm.save_pretrained(save_model_path)
                        torch.save(
                            model.projector.state_dict(),
                            f"{save_model_path}_projector.pth",
                        )
                        torch.save(
                            model.dino.state_dict(), f"{save_model_path}_dino.pth"
                        )
                        torch.save(
                            model.siglip.state_dict(), f"{save_model_path}_siglip.pth"
                        )
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
        dataset_dir="libero/libero/datasets/libero_spatial",
        llm_model_name="meta-llama/Llama-3.2-1B",
        batch_size=2,
        num_epochs=5,
        lr_rate=1e-5,
        patience=3,
        eval_interval=1000,
        num_workers=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_model_path="checkpoints/libero_spatial",
        gradient_accumulation_steps=2,
        lora_r=8,
        lora_alpha=32,
        val_demos=5,
    )
