import matplotlib.pyplot as plt
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA


def get_pca_map(features, side_len):
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)

    for i in range(3):
        v_min, v_max = pca_features[:, i].min(), pca_features[:, i].max()
        if v_max - v_min > 1e-5:
            pca_features[:, i] = (pca_features[:, i] - v_min) / (v_max - v_min)

    return pca_features.reshape(side_len, side_len, 3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = (
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(device).eval()
    )

    siglip_model = (
        timm.create_model("vit_so400m_patch14_siglip_384", pretrained=True)
        .to(device)
        .eval()
    )

    patch_size = 14
    img_size = 384
    num_patches_side = img_size // patch_size
    img_raw = Image.open("caption.jpg").convert("RGB")

    dino_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dino_input = dino_transform(img_raw).unsqueeze(0).to(device)

    siglip_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    siglip_input = siglip_transform(img_raw).unsqueeze(0).to(device)

    with torch.no_grad():
        dino_input_378 = F.interpolate(
            dino_input, size=(378, 378), mode="bilinear", align_corners=False
        )
        dino_features_dict = dino_model.forward_features(dino_input_378)
        dino_patchs = dino_features_dict["x_norm_patchtokens"].squeeze(0).cpu().numpy()

        siglip_out = siglip_model.forward_features(siglip_input)
        siglip_patches = siglip_out.squeeze(0).cpu().numpy()

    dino_pca_img = get_pca_map(dino_patchs, num_patches_side)
    siglip_pca_img = get_pca_map(siglip_patches, num_patches_side)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_raw.resize((img_size, img_size)))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(dino_pca_img, interpolation="bilinear")
    axes[1].set_title("DINOv2 PCA Feature Map")
    axes[1].axis("off")

    axes[2].imshow(siglip_pca_img, interpolation="bilinear")
    axes[2].set_title("SigLIP PCA Feature Map")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
