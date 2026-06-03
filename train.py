import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from action_tokenizer import ActionTokenizer
from dataloader import get_dataloader
from model import VLAModel


def train(llm_model_name, batch_size, num_epochs, lr_rate, device):
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
    for i, batch in enumerate(train_loader):
        print(f"--- Batch(train_loader) {i} ---")
        for k, v in batch.items():
            print(f"{k}: {tuple(v.shape)} ({v.dtype})")
        break

    val_loader = get_dataloader(tokenizer, at, batch_size=4, split="train")
    for i, batch in enumerate(train_loader):
        print(f"--- Batch(train_loader) {i} ---")
        for k, v in batch.items():
            print(f"{k}: {tuple(v.shape)} ({v.dtype})")
        break


if __name__ == "__main__":
    llm_model_name = "meta-llama/Llama-3.2-1B"
    batch_size = 2
    num_epochs = 1
    lr_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(llm_model_name, batch_size, num_epochs, lr_rate, device)
