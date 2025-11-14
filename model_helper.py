import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "fastapi_server/saved_model.pth"   # ✅ correct full path

# ✅ Load base ResNet50
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)  # <--- change number of classes if needed

# ✅ Load state dict safely
state_dict = torch.load(MODEL_PATH, map_location="cpu")

# ✅ Fix key mismatch: remove "model." prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("model.", "")
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# ✅ Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    labels = ["Dent", "Scratch", "No Damage"]  # ✅ change to real labels
    return labels[predicted.item()]


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return int(predicted.item())
