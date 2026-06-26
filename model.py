import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class VLAModel(nn.Module):
    def __init__(
        self,
        llm_model_name,
        checkpoint_path=None,
        dino_checkpoint_path=None,
        siglip_checkpoint_path=None,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        # Vision encoder
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(
            device
        )
        self.siglip = timm.create_model(
            "vit_so400m_patch14_siglip_384",
            pretrained=True,
            num_classes=0,
        ).to(device)

        # Get dimemtins
        dino_dim = self.dino.num_features  # 1024
        siglip_dim = self.siglip.num_features  # 1152
        vision_dim = dino_dim + siglip_dim  # 2176

        # Language model (fp32 master weights; fp16 is applied via autocast)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, low_cpu_mem_usage=True
        ).to(device)
        llm_dim = self.llm.config.hidden_size  # 2048

        # Projection layers (fp32 master weights)
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim)
        ).to(device=device)

        # Per-encoder image normalization stats. DINOv2 was pretrained with
        # ImageNet mean/std; SigLIP was pretrained with mean=std=0.5. Applying
        # the wrong (or no) normalization silently degrades the vision features
        # because the input distribution no longer matches what the frozen
        # backbones saw during pretraining. We expect the dataloader to hand us
        # raw [0, 1] RGB tensors and do the normalization here so each encoder
        # sees its native input space.
        self.register_buffer(
            "_dino_mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_siglip_mean",
            torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_siglip_std",
            torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1),
            persistent=False,
        )

        # Load checkpoint (all components at once)
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            # Load to CPU first to avoid doubling GPU memory during state_dict copy
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict)
            del state_dict
            torch.cuda.empty_cache()
            print("Checkpoint loaded.")

        # Load fine-tuned vision encoder weights separately if provided
        if dino_checkpoint_path is not None:
            print(f"Loading DINO checkpoint from {dino_checkpoint_path}")
            sd = torch.load(dino_checkpoint_path, map_location="cpu")
            self.dino.load_state_dict(sd)
            del sd
            torch.cuda.empty_cache()
            print("DINO checkpoint loaded.")

        if siglip_checkpoint_path is not None:
            print(f"Loading SigLIP checkpoint from {siglip_checkpoint_path}")
            sd = torch.load(siglip_checkpoint_path, map_location="cpu")
            self.siglip.load_state_dict(sd)
            del sd
            torch.cuda.empty_cache()
            print("SigLIP checkpoint loaded.")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 384, 384), raw RGB in [0, 1] (no normalization applied)
        returns: (B, N, vision_dim)  N = 27 * 27 = 729 patches
        """
        # DINOv2 requires H, W divisible by patch_size=14 -> resize to 378,
        # then apply ImageNet normalization (DINOv2's pretraining stats).
        dino_input = F.interpolate(
            images, size=(378, 378), mode="bilinear", align_corners=False
        )
        dino_input = (dino_input - self._dino_mean) / self._dino_std
        dino_out = self.dino.forward_features(dino_input)
        dino_feats = dino_out["x_norm_patchtokens"]  # (B, 729, 1024)

        # SigLIP runs natively at 384 (floor(384/14)=27 patches per side) and
        # was pretrained with mean=std=0.5 normalization.
        siglip_input = (images - self._siglip_mean) / self._siglip_std
        siglip_feats = self.siglip.forward_features(siglip_input)  # (B, 729, 1152)

        vision_feats = torch.cat([dino_feats, siglip_feats], dim=-1)  # (B, 729, 2176)
        return vision_feats

    def _build_inputs(self, images, input_ids, attention_mask):
        # Encode image -> projection
        vision_feats = self.encode_image(images)  # (B, N, 2176)
        vision_embeds = self.projector(vision_feats)  # (B, N, llm_dim)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, llm_dim)

        # Concat image embeddings and text embeddings
        input_embeds = torch.cat(
            [vision_embeds, text_embeds], dim=1
        )  # (B, N+T, llm_dim)

        # Extend attantion
        B, N, _ = vision_embeds.shape
        vision_mask = torch.ones(
            B, N, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([vision_mask, attention_mask], dim=1)

        return input_embeds, full_mask, N

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        images: (B, 3, 384, 384)
        input_ids: (B, T)
        attention_mask: (B, T)
        labels: (B, T)
        """

        input_embeds, full_mask, N = self._build_inputs(
            images, input_ids, attention_mask
        )
        if labels is not None:
            B = images.shape[0]
            vision_labels = torch.full(
                (B, N), -100, dtype=labels.dtype, device=labels.device
            )
            full_labels = torch.cat([vision_labels, labels], dim=1)  # (B, N+T)
        else:
            full_labels = None

        # Pass inputs to LLM
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return outputs

    def generate(self, images, input_ids, attention_mask, max_new_tokens=7):
        """
        For inference

        images: (B, 3, 384, 384)
        returns: token ids (B, max_new_tokens)
        """
        input_embeds, full_mask, _ = self._build_inputs(
            images, input_ids, attention_mask
        )

        # Pass pad_token_id explicitly so HF doesn't log
        # "Setting `pad_token_id` to `eos_token_id`..." on every call.
        eos_token_id = self.llm.config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]

        output_ids = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=full_mask,
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
    model = VLAModel("meta-llama/Llama-3.2-1B", device=device)

    images = torch.randn(2, 3, 384, 384).to(device)
    input_ids = torch.randint(0, 1000, (2, 10)).to(device)
    attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
    labels = torch.randint(0, 1000, (2, 10)).to(device)

    # Training
    out = model(images, input_ids, attention_mask, labels=labels)
    print("loss:", out.loss)
    print("logits:", out.logits.shape)  # (2, N+10, vocab_size)

    # Inference
    generated = model.generate(images, input_ids, attention_mask)
    print("generated:", generated.shape)
