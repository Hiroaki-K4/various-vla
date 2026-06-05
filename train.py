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
    count = 0

    for batch_data in val_loader:
        images = batch_data["image"].to(device)
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(images, input_ids, attention_mask, labels)
            loss = outputs.loss

        if loss is not None:
            total_loss += loss.item()
            count += 1
        else:
            print(f"Warning: loss is None in evaluate")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"images shape: {images.shape}")
            print(f"outputs keys: {dir(outputs)}")

    if count == 0:
        print("Warning: No valid batches processed in evaluate")
        return float("inf")

    model.train()
    return total_loss / count


def train(
    llm_model_name,
    batch_size,
    num_epochs,
    lr_rate,
    patience,
    eval_interval,
    device,
    save_model_path,
    gradient_accumulation_steps=1,
):
    print("Loading models...")
    model = VLAModel(llm_model_name, device=device)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model.llm = get_peft_model(model.llm, lora_config)

    model.llm.enable_input_require_grads()
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    at = ActionTokenizer(tokenizer, n_bins=256)

    train_loader = get_dataloader(tokenizer, at, batch_size=4, split="train")
    val_loader = get_dataloader(tokenizer, at, batch_size=4, split="val")

    # Parameters for early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    print("Starting Training...")

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        i = 0
        for batch_data in pbar:
            images = batch_data["image"].to(device)
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images, input_ids, attention_mask, labels)
                loss = outputs.loss / gradient_accumulation_steps

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: invalid loss at step {i} (loss={loss})")
                i += 1
                continue

            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Clear memory cache

            print("loss:", loss.item() * gradient_accumulation_steps)
            input()

            pbar.set_postfix({"train_loss": loss.item() * gradient_accumulation_steps})

            i += 1
            if i % eval_interval == 0 and (i % gradient_accumulation_steps == 0):
                torch.cuda.empty_cache()  # Clear cache before evaluation
                val_loss = evaluate(model, val_loader, device)
                print(f"\nStep {i} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model.llm.save_pretrained(save_model_path)
                    # TODO: Think about which model to save
                    print("New best model saved!")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    return


if __name__ == "__main__":
    llm_model_name = "meta-llama/Llama-3.2-1B"
    batch_size = 2
    num_epochs = 1
    lr_rate = 1e-5
    patience = 3
    eval_interval = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_model_path = "vla_model"
    gradient_accumulation_steps = 2
    train(
        llm_model_name,
        batch_size,
        num_epochs,
        lr_rate,
        patience,
        eval_interval,
        device,
        save_model_path,
        gradient_accumulation_steps,
    )
