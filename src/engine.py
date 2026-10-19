REPO = 'palmyz0/brain-tumor-resnet50-finetune-with-optuna'
FILENAME = 'model.pth'
LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    import torch
    from torch import nn
    from torchvision import models
    from huggingface_hub import hf_hub_download
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(0.48), nn.Linear(512,4))
    class Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            return self.backbone(x)
    model = Wrapper(model)
    path = hf_hub_download(REPO, FILENAME)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def transform_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

def predict(image, threshold=0.35):
    import torch
    from src.validation import threshold_value
    with torch.inference_mode():
        scores = load_model()(transform_image(image)).softmax(dim=1)[0].tolist()
    probabilities = dict(zip(LABELS, [round(float(s),6) for s in scores]))
    label = max(probabilities, key=probabilities.get)
    return {"label": label, "score": probabilities[label], "scores": probabilities,
            "review_required": probabilities[label] < threshold_value(threshold),
            "model": REPO}, None
