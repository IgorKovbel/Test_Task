from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch

try:
    from .train import transform, DEVICE, model, class_names
except ImportError:
    from train import transform, DEVICE, model, class_names

model = model.to(DEVICE)
model.load_state_dict(torch.load("./classifier/best_model.pt", map_location=torch.device(DEVICE)))

def run_inference(image):
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    
    return class_names[pred.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--image_path", type=str, help="Path to the image file")

    args = parser.parse_args()
    image = Image.open(args.image_path).convert('RGB')

    pred = run_inference(image)

    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    plt.title(f'Prediction: {pred}')
    plt.axis('off')
    plt.show()
