# predict.py

import torch
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y, device

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_batch(model, list_of_image_paths):
    model = model.to(device)
    model.eval()
    images = [load_image(p) for p in list_of_image_paths]
    batch = torch.cat(images, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)

    return preds.cpu().tolist()
