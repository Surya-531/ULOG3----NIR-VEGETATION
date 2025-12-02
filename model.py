import os, cv2, torch, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "best_unet_resnet34_vegseg.pth"  # fixed name

# -------- load checkpoint --------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model = smp.Unet(
    encoder_name=ckpt["encoder_name"],  # "resnet34"
    in_channels=ckpt["in_channels"],    # 1
    classes=ckpt["classes"]             # 1
).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

IMG_SIZE = ckpt.get("img_size", 256)
mean = ckpt.get("normalize", {}).get("mean", [0.0])
std  = ckpt.get("normalize", {}).get("std",  [1.0])
THRESH = ckpt.get("threshold", 0.5)

# Preprocess (must match training normalization, NO random augmentations)
infer_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=tuple(mean), std=tuple(std)),
    ToTensorV2()
])

def preprocess_gray(img):
    aug = infer_tf(image=img)
    t = aug["image"].unsqueeze(0).to(DEVICE)  # [1,1,H,W]
    return t

@torch.no_grad()
def predict_mask(nir_img_uint8):
    t = preprocess_gray(nir_img_uint8)
    logits = model(t)              
    prob   = torch.sigmoid(logits) 
    pred   = (prob >= THRESH).float()[0,0].cpu().numpy()  
    return (pred*255).astype(np.uint8)                    

# -------- Single image inference --------
def infer_single(nir_path):
    img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    mask255 = predict_mask(img)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Input NIR Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask255, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()

# Example usage
infer_single("DATASET/NIR/IMG_210203_140952_0202_NIR.TIF")
infer_single("DATASET/NIR/IMG_210203_141147_0223_NIR.TIF")
infer_single("DATASET/NIR/IMG_210203_141423_0260_NIR.TIF")
