import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class VLAModel(nn.Module):
    def __init__(self, llm_model_name, projector_path=None, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print("vision_dim:", vision_dim)

        # Language model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        llm_dim = self.llm.config.hidden_size  # 2048
        print("llm_dim:", llm_dim)

        # Projection layers
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim)
        ).to(device=device, dtype=torch.bfloat16)
        if projector_path is not None:
            self.projector.load_state_dict(
                torch.load(projector_path, map_location=device)
            )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 384, 384)  # SigLIP native size
        returns: (B, N, vision_dim)  N = 27 * 27 = 729 patches
        """
        with torch.no_grad():
            # DINOv2 requires H, W divisible by patch_size=14 -> resize to 378
            dino_input = F.interpolate(
                images, size=(378, 378), mode="bilinear", align_corners=False
            )
            dino_out = self.dino.forward_features(dino_input)
            dino_feats = dino_out["x_norm_patchtokens"]  # (B, 729, 1024)

            # SigLIP runs natively at 384 (floor(384/14)=27 patches per side)
            siglip_feats = self.siglip.forward_features(images)  # (B, 729, 1152)

        vision_feats = torch.cat([dino_feats, siglip_feats], dim=-1)  # (B, 729, 2176)
        return vision_feats.to(torch.bfloat16)

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

        # 1. Encode image -> projection
        vision_feats = self.encode_image(images)  # (B, N, 2176)
        vision_embeds = self.projector(vision_feats)  # (B, N, 2048)

        # 2. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, 2048)

        # 3. Concat image embeddings and text embeddings
        input_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # (B, N+T, 2048)

        # Extend attantion
        B, N, _ = vision_embeds.shape
        vision_mask = torch.ones(
            B, N, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # 4. Pass inputs to LLM
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=full_mask,
            labels=labels,
        )
        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLAModel("meta-llama/Llama-3.2-1B", device=device)

    images = torch.randn(2, 3, 384, 384).to(device)
    input_ids = torch.randint(0, 1000, (2, 10)).to(device)
    attention_mask = torch.ones(2, 10, dtype=torch.long).to(device)

    out = model(images, input_ids, attention_mask)
    print(out.logits.shape)  # (2, N+10, vocab_size)
