import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def _tensor_to_pil(image_tensor):
    """Convert (3, H, W) float32 [0,1] tensor to PIL Image"""
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


class PlamoVLAModel(nn.Module):
    """
    VLA model based on Plamo-2.1-2B-VL vision-language model.
    Plamo already provides integrated vision and language components.

    Action prediction approach:
    - Uses tokenized actions: continuous values are discretized into tokens
    - LLM predicts action tokens directly as part of the sequence
    - No separate action head needed - LLM vocabulary includes action tokens
    - Post-training: action tokens are converted back to continuous values via ActionTokenizer

    Key architecture details:
    - text_config.hidden_size: 2048
    - vision_config.image_feature_size: 1152
    - image_proj: Already integrated MLPImageProjector (1152 -> 2048)
    - vision_model: SiglipVisionTransformer (384x384 input)
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
            dtype=torch.float32,
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
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass for training.

        Args:
            images: (B, 3, H, W) or (B, V, 3, H, W) - Raw RGB images in [0, 1]
                    V is number of camera views (e.g., multi-view input)
            input_ids: (B, T) - Tokenized text input
            attention_mask: (B, T) - Attention mask for text
            labels: (B, T) - Target token IDs for action prediction

        Returns:
            outputs with loss and per-sample loss for multi-task learning
        """

        # Handle multi-view format: (B, V, 3, H, W) → (B, 3, H, W)
        # For now, use only first view (V=1 is default)
        if images.dim() == 5:
            images = images[:, 0]  # Take first view only

        batch_size = images.shape[0]

        # Process images with Plamo's processor
        # Note: processor expects PIL Images (not tensors) and text as lists
        pil_images = [_tensor_to_pil(images[i]) for i in range(batch_size)]
        text_list = [""] * batch_size  # Empty text (image-only input)

        processed = self.processor(
            images=pil_images,
            text=text_list,
            return_tensors="pt",
        )

        # Plamo's forward pass directly handles vision + text combination
        # Get pixel values from processor
        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

            # Forward through model with both vision and text
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,  # Compute loss manually
            )
        else:
            # Fallback if processor doesn't return pixel_values
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )

        # Compute per-sample loss for multi-task learning
        if labels is not None:
            logits = outputs.logits  # (B, seq_len, vocab_size)
            vocab_size = logits.shape[-1]

            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # Compute per-token loss
            token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")

            # Reshape back to (B, T)
            token_loss = token_loss.view(batch_size, -1)

            # Create loss mask (valid tokens only, -100 means ignore)
            loss_mask = (shift_labels.view(batch_size, -1) != -100).float()

            # Per-sample loss
            per_sample_loss = (token_loss * loss_mask).sum(dim=1) / (
                loss_mask.sum(dim=1) + 1e-8
            )
            batch_loss = per_sample_loss.mean()

            outputs.loss = batch_loss
            outputs.per_sample_loss = per_sample_loss

        return outputs

    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 7,
    ):
        """
        Generate action tokens given image and text input.

        Args:
            images: (B, 3, H, W) or (B, V, 3, H, W)
            input_ids: (B, T)
            attention_mask: (B, T)
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated token IDs (B, max_new_tokens)
        """

        # Handle multi-view format: (B, V, 3, H, W) → (B, 3, H, W)
        # For now, use only first view (V=1 is default)
        if images.dim() == 5:
            images = images[:, 0]  # Take first view only

        batch_size = images.shape[0]

        # Process images - convert tensor to PIL Image
        pil_images = [_tensor_to_pil(images[i]) for i in range(batch_size)]
        text_list = [""] * batch_size  # Empty text (image-only input)

        processed = self.processor(
            images=pil_images,
            text=text_list,
            return_tensors="pt",
        )

        pixel_values = processed.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

            # Generate
            eos_token_id = self.model.config.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]

            output_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=eos_token_id,
            )
        else:
            # Fallback
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        return output_ids

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlamoVLAModel(device=device)

    images = torch.randn(2, 3, 384, 384).to(device)
    input_ids = torch.randint(0, 1000, (2, 10)).to(device)
    attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
    labels = torch.randint(0, 1000, (2, 10)).to(device)

    # Training
    out = model(images, input_ids, attention_mask, labels=labels)
    print("loss:", out.loss)
    print("logits:", out.logits.shape)

    # Inference
    generated = model.generate(images, input_ids, attention_mask)
    print("generated:", generated.shape)
