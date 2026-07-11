import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor


class PlamoVLAModel(nn.Module):
    """
    VLA model based on Plamo-2.1-2B-VL.

    Uses processor to handle image + text → input_ids + pixel_values conversion.
    Processor automatically inserts image_token_id into input_ids.

    Action prediction:
    - Prompt + action tokens are fed to processor
    - Processor handles image_token_id insertion
    - Model predicts next tokens (including action tokens)
    """

    def __init__(
        self,
        plamo_model_name: str = "pfnet/plamo-2.1-2b-vl",
        checkpoint_path=None,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.plamo_model_name = plamo_model_name

        # Load Plamo VL model
        print(f"Loading {plamo_model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            plamo_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(
            plamo_model_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        # Load checkpoint if provided
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)
            del state_dict
            torch.cuda.empty_cache()
            print("Checkpoint loaded.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass for training.

        Args:
            input_ids: (B, T) - Processed by collate_fn (with image_token inserted)
            attention_mask: (B, T) - From processor
            pixel_values: (total_tiles, C, H, W) - From processor (not batched)
            labels: (B, T) - Target tokens for loss (use -100 to ignore)

        Returns:
            Model outputs (includes loss when labels provided)
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 7,
    ):
        """
        Generate action tokens.

        Args:
            input_ids: (B, T) - Processed by collate_fn (with image_token inserted)
            attention_mask: (B, T) - From processor
            pixel_values: (total_tiles, C, H, W) - From processor
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated token IDs only (B, max_new_tokens) - excludes input_ids
        """

        eos_token_id = self.model.config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]

        input_len = input_ids.shape[1]

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

        # Ensure we return only new tokens (not prompt)
        # Some models return [input_ids, new_tokens], others return [new_tokens]
        if output_ids.shape[1] > input_len:
            output_ids = output_ids[:, input_len:]

        return output_ids

    def save_checkpoint(self, path):
        import os

        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            raise ValueError(
                "Model must be a PeftModel (LoRA adapter) to save. "
                "Ensure get_peft_model() was applied during initialization."
            )

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        # Saves: adapter_config.json + adapter_model.safetensors (PEFT v0.6+) or .bin
        print(f"LoRA adapter saved to {path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlamoVLAModel(device=device)

    # Test with proper input from processor
    input_ids = torch.randint(0, 1000, (2, 100)).to(device)
    attention_mask = torch.ones(2, 100, dtype=torch.long).to(device)
    pixel_values = torch.randn(200, 3, 384, 384).to(device)  # (total_tiles, C, H, W)
    labels = torch.randint(0, 1000, (2, 100)).to(device)

    # Forward
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    )
    print("loss:", out.loss)
    print("logits:", out.logits.shape)

    # Generate
    generated = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values
    )
    print("generated:", generated.shape)
