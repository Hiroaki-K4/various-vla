import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class PlamoVLAModel(nn.Module):
    """
    VLA model based on Plamo-2.1-2B-VL vision-language model.
    Plamo already provides integrated vision and language components.

    Action prediction approach:
    - Uses tokenized actions: continuous values are discretized into tokens
    - LLM predicts action tokens directly as part of the sequence
    - No separate action head needed - LLM vocabulary includes action tokens
    - Post-training: action tokens are converted back to continuous values via ActionTokenizer
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
            torch_dtype=torch.float32,
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
            images: (B, 3, H, W) - Raw RGB images in [0, 1]
            input_ids: (B, T) - Tokenized text input
            attention_mask: (B, T) - Attention mask for text
            labels: (B, T) - Target token IDs for action prediction

        Returns:
            outputs with loss and per-sample loss for multi-task learning
        """

        # Process images with Plamo's processor
        # Note: Plamo expects images in its native format
        processed = self.processor(
            images=images,
            return_tensors="pt",
        )

        # Get image features from Plamo
        image_features = processed.get("image_features")
        if image_features is None:
            # Fallback: run through Plamo's vision encoder
            # This depends on Plamo's internal structure
            with torch.no_grad():
                image_embeds = self.model.vision_model(images)
        else:
            image_embeds = image_features.to(self.device)

        # Get text embeddings
        text_embeds = self.model.get_input_embeddings()(input_ids)

        # Combine image and text embeddings
        # For Plamo, we need to follow its expected input format
        combined_embeds = self._combine_modalities(image_embeds, text_embeds)

        # Forward through language model
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=None,  # Compute loss manually
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
            B = images.shape[0]
            token_loss = token_loss.view(B, -1)

            # Create loss mask (valid tokens only)
            loss_mask = (shift_labels.view(B, -1) != -100).float()

            # Per-sample loss
            per_sample_loss = (token_loss * loss_mask).sum(dim=1) / (
                loss_mask.sum(dim=1) + 1e-8
            )
            batch_loss = per_sample_loss.mean()

            outputs.loss = batch_loss
            outputs.per_sample_loss = per_sample_loss

        return outputs

    def _combine_modalities(self, image_embeds, text_embeds):
        """
        Combine image and text embeddings.
        The exact method depends on Plamo's architecture.
        """
        # Simple concatenation (may need adjustment based on Plamo's design)
        if image_embeds.dim() == 2:
            # image_embeds: (B, vision_dim)
            image_embeds = image_embeds.unsqueeze(1)  # (B, 1, vision_dim)

        # Project image embeddings to text embedding dimension if needed
        if image_embeds.shape[-1] != text_embeds.shape[-1]:
            # Would need a projection layer here
            pass

        combined = torch.cat([image_embeds, text_embeds], dim=1)
        return combined

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
            images: (B, 3, H, W)
            input_ids: (B, T)
            attention_mask: (B, T)
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated token IDs (B, max_new_tokens)
        """

        # Process images
        processed = self.processor(
            images=images,
            return_tensors="pt",
        )

        image_embeds = processed.get("image_features")
        if image_embeds is None:
            with torch.no_grad():
                image_embeds = self.model.vision_model(images)
        else:
            image_embeds = image_embeds.to(self.device)

        # Get text embeddings
        text_embeds = self.model.get_input_embeddings()(input_ids)

        # Combine
        combined_embeds = self._combine_modalities(image_embeds, text_embeds)

        # Generate
        eos_token_id = self.model.config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]

        output_ids = self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=eos_token_id,
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
