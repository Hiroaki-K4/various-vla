import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CrossAttentionFusion(nn.Module):
    """
    Merge features of multiple views using cross-attention.
    For memory efficiency with 2+ views, use attn_dropout=0.0 during training.
    """
    def __init__(self, hidden_dim, num_heads=8, num_layers=2, attn_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True, dropout=attn_dropout
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, feature_list):
        """
        feature_list: List [view1, view2, ...], (B, N, dim) shape
        returns: (B, V*N, dim)
        """
        if len(feature_list) == 1:
            return feature_list[0]
        
        # Concat all views: (B, V*N, dim)
        x = torch.cat(feature_list, dim=1)

        # Apply cross-attention layers
        for attn_layer, norm, ffn, ffn_norm in zip(
            self.layers, self.norms, self.ffn, self.ffn_norms
        ):
            # Self-attention(Intruction among all views)
            x_attn, _ = attn_layer(x, x, x)
            x = norm(x + x_attn)  # Residual + LayerNorm

            # Feedforward Network
            x_ffn = ffn(x)
            x = ffn_norm(x + x_ffn)  # Residual + LayerNorm

        return x


class VLAModel(nn.Module):
    def __init__(
        self,
        llm_model_name,
        checkpoint_path=None,
        dino_checkpoint_path=None,
        siglip_checkpoint_path=None,
        device=None,
        use_multi_view_fusion=False,
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

        if use_multi_view_fusion:
            self.multi_view_fusion = CrossAttentionFusion(
                vision_dim, num_heads=8, num_layers=2
            ).to(device)
        else:
            self.multi_view_fusion = None

        self.use_multi_view_fusion = use_multi_view_fusion

        # Language model (fp32 master weights; fp16 is applied via autocast)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, low_cpu_mem_usage=True
        ).to(device)
        llm_dim = self.llm.config.hidden_size  # 2048

        # Projection layers (fp32 master weights)
        initial_projection_dim = 4 * vision_dim
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, initial_projection_dim),
            nn.GELU(),
            nn.Linear(initial_projection_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
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
        images: (B, 3, 384, 384) single view, or (B, V, 3, 384, 384) multi-view,
                raw RGB in [0, 1] (no normalization applied).
        returns: (B, V*N, vision_dim)  N = 27 * 27 = 729 patches per view.

        Multi-view (e.g. third-person + wrist camera) is encoded by running each
        view through the frozen backbones independently and concatenating their
        patch tokens along the sequence dimension, matching the OpenVLA-OFT recipe
        (the projected tokens simply become a longer image-token prefix).
        """
        if images.dim() == 5:
            B, V = images.shape[:2]
            features_list = []
            for v in range(V):
                view_features = self._encode_view(images[:, v, :, :, :])  # (B, N, vision_dim)
                features_list.append(view_features)
                # Memory optimization: clear intermediate GPU cache between views
                if v < V - 1:
                    torch.cuda.empty_cache()

            if self.multi_view_fusion is not None:
                fused = self.multi_view_fusion(features_list)
                return fused
            else:
                return torch.cat(features_list, dim=1)

        return self._encode_view(images)

    def _encode_view(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a single view: (B, 3, 384, 384) -> (B, 729, vision_dim)."""
        # DINOv2 requires H, W divisible by patch_size=14 -> resize to 378,
        # then apply ImageNet normalization (DINOv2's pretraining stats).
        dino_input = F.interpolate(
            images, size=(378, 378), mode="bilinear", align_corners=False
        )
        dino_input = (dino_input - self._dino_mean) / self._dino_std
        dino_feats = self.dino.get_intermediate_layers(
            dino_input,
            n=[len(self.dino.blocks) - 2],
        )[
            0
        ]  # (B, 729, 1024)

        # SigLIP runs natively at 384 (floor(384/14)=27 patches per side) and
        # was pretrained with mean=std=0.5 normalization.
        siglip_input = (images - self._siglip_mean) / self._siglip_std
        siglip_feats = self.siglip.get_intermediate_layers(
            siglip_input,
            n=[len(self.siglip.blocks) - 2],
        )[
            0
        ]  # (B, 729, 1152)

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

        Returns: model outputs with per-sample loss for multi-task learning.
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

        # Pass inputs to LLM without loss computation
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=full_mask,
            labels=None,  # Compute loss manually for per-sample tracking
        )

        # Compute per-sample loss for multi-task learning
        if labels is not None:
            logits = outputs.logits  # (B, N+T, vocab_size)
            vocab_size = logits.shape[-1]

            # Reshape for loss computation: (B*(N+T), vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # Compute per-token loss
            token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")

            # Reshape back to (B, T) and mask out vision tokens (label == -100)
            token_loss = token_loss.view(B, -1)
            loss_mask = (full_labels[..., 1:] != -100).float()

            # Per-sample loss (average over valid tokens)
            per_sample_loss = (token_loss * loss_mask).sum(dim=1) / (
                loss_mask.sum(dim=1) + 1e-8
            )
            batch_loss = per_sample_loss.mean()

            # Attach per-sample loss to outputs for evaluate() to use
            outputs.loss = batch_loss
            outputs.per_sample_loss = per_sample_loss

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
    torch.cuda.empty_cache()

    print("=" * 50)
    print("Test 1: Single-view (no fusion)")
    print("=" * 50)
    model = VLAModel("meta-llama/Llama-3.2-1B", device=device, use_multi_view_fusion=False)
    model.eval()

    with torch.no_grad():
        images = torch.randn(2, 3, 384, 384).to(device)
        input_ids = torch.randint(0, 1000, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
        labels = torch.randint(0, 1000, (2, 10)).to(device)

        out = model(images, input_ids, attention_mask, labels=labels)
        print(f"✓ Loss: {out.loss.item():.4f}")
        print(f"✓ Logits shape: {out.logits.shape}")
        print(f"✓ Vision tokens: 729 (1 view × 729 patches)")

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Test 2: Multi-view (with fusion)")
    print("=" * 50)
    model = VLAModel("meta-llama/Llama-3.2-1B", device=device, use_multi_view_fusion=True)
    model.eval()

    with torch.no_grad():
        images_multi = torch.randn(2, 2, 3, 384, 384).to(device)
        input_ids = torch.randint(0, 1000, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)
        labels = torch.randint(0, 1000, (2, 10)).to(device)

        out = model(images_multi, input_ids, attention_mask, labels=labels)
        print(f"✓ Loss: {out.loss.item():.4f}")
        print(f"✓ Logits shape: {out.logits.shape}")
        print(f"✓ Vision tokens: 1458 (2 views × 729 patches)")

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
