import torch
import torchvision.transforms as T
from PIL import Image

devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(devide)

model.eval()

patch_size = 14
img_size = 448
num_patches_side = img_size // patch_size

transform = T.Compose(
    [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img_raw = Image.open("").convert("RGB")

# TODO: https://gemini.google.com/app/5363bbc8a7d98a80
