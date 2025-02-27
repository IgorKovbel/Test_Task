import argparse
from PIL import Image
from classifier.inference import run_inference
from NER.inference import run_ner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--image_path", type=str)

    args = parser.parse_args()

    image = Image.open(args.image_path).convert('RGB')
    entities = run_ner(args.text)
    pred = run_inference(image)

    print(pred in entities)

