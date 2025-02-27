import warnings
import torch
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import argparse


warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
torch.set_printoptions(sci_mode=False)

model = AutoModelForTokenClassification.from_pretrained('./NER/trained_ner_model') 
tokenizer = AutoTokenizer.from_pretrained('./NER/trained_ner_model')

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy='first'
)

def run_ner(text):
    predictions = ner_pipeline(text)
    extracted_entities = [item["word"] for item in predictions if item['entity_group'] == 'ANIM']

    return extracted_entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)

    args = parser.parse_args()
    entities = run_ner(args.text)
    print(entities)