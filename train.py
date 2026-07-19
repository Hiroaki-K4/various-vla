import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer

from action_tokenizer import ActionTokenizer
from dataloader import get_dataloader
from model import VLAModel


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    batch_count = 0

    for batch_data in val_loader:
        batch_count += 1
        images = batch_data["image"].to(device)
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss

        if loss is not None:
            total_loss += loss.item()

    model.train()
    avg_loss = total_loss / batch_count if batch_count > 0 else float("inf")
    return avg_loss


def train(
    llm_model_name,
    batch_size,
    num_epochs,
    lr_rate,
    patience,
    eval_interval,
    num_workers,
    device,
    save_model_path,
    gradient_accumulation_steps=1,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.0,
):
    print("Loading models...")
    model = VLAModel(llm_model_name, device=device)
    # Apply LoRA to all linear projections in the Llama transformer blocks
    # (attention q/k/v/o + MLP gate/up/down). This is the standard recipe used
    # by OpenVLA / LLaVA and gives noticeably more capacity than the default
    # q_proj+v_proj-only setup.
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

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    at = ActionTokenizer(tokenizer, n_bins=256)

    train_loader = get_dataloader(
        tokenizer, at, batch_size=batch_size, split="train", num_workers=num_workers
    )
    val_loader = get_dataloader(
        tokenizer, at, batch_size=batch_size, split="val", num_workers=num_workers
    )


    # Parameters for early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    # Frozen vision encoders stay in eval mode (no dropout / norm-stat updates)
    model.dino.eval()
    model.siglip.eval()
    print("Starting Training...")

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch_data in enumerate(pbar):
            images = batch_data["image"].to(device)
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)

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
                        model.llm.save_pretrained(save_model_path)
                        torch.save(
                            model.projector.state_dict(),
                            f"{save_model_path}_projector.pth",
                        )
                        torch.save(
                            model.dino.state_dict(),
                            f"{save_model_path}_dino.pth",
                        )
                        torch.save(
                            model.siglip.state_dict(),
                            f"{save_model_path}_siglip.pth",
                        )
                        print("New best model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    print(f"Best Val Loss: {best_val_loss:.4f}")
                    return best_val_loss

    return best_val_loss


if __name__ == "__main__":
    llm_model_name = "meta-llama/Llama-3.2-3B"
    batch_size = 2
    num_epochs = 5
    lr_rate = 1e-5
    patience = 5
    eval_interval = 1000
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_model_path = "best_vla_model"
    gradient_accumulation_steps = 2
    lora_r = 32
    lora_alpha = 128
    train(
        llm_model_name,
        batch_size,
        num_epochs,
        lr_rate,
        patience,
        eval_interval,
        num_workers,
        device,
        save_model_path,
        gradient_accumulation_steps,
        lora_r,
        lora_alpha,
    )
