import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments
from evaluate import load
seqeval = load("seqeval")

label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30
}
id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)

class NERJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length=128):
        super().__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                tokens = data["tokens"]
                tags = data["ner_tags"]
                self.samples.append((tokens, tags))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, numeric_tags = self.samples[idx]

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)

        aligned_labels = []
        for w_id in word_ids:
            if w_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(numeric_tags[w_id])

        encoding["labels"] = torch.tensor(aligned_labels, dtype=torch.long)

        return {k: v.squeeze() for k, v in encoding.items()}
    

def compute_metrics(p):

    logits, labels = p
    predictions = logits.argmax(axis=-1)

    true_labels = []
    pred_labels = []
    for pred_row, lab_row in zip(predictions, labels):
        tmp_true = []
        tmp_pred = []
        for p_i, l_i in zip(pred_row, lab_row):
            if l_i == -100:
                continue
            tmp_true.append(id2label[l_i])
            tmp_pred.append(id2label[p_i])
        true_labels.append(tmp_true)
        pred_labels.append(tmp_pred)

    results = seqeval.compute(predictions=pred_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

model_name_or_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    num_labels=NUM_LABELS,
    id2label=id2label, 
    label2id=label2id
)

train_dataset = NERJsonlDataset(
    jsonl_path=Path("C:/Work/Test_Task/multinerd/train/train_en.jsonl"),
    tokenizer=tokenizer,
    max_length=128
)
val_dataset = NERJsonlDataset(
    jsonl_path=Path("C:/Work/Test_Task/multinerd/val/val_en.jsonl"),
    tokenizer=tokenizer,
    max_length=128
)
test_dataset = NERJsonlDataset(
    jsonl_path=Path("C:/Work/Test_Task/multinerd/test/test_en.jsonl"),
    tokenizer=tokenizer,
    max_length=128
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="output_ner",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="no",
    logging_steps=10,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    trainer.train()

    trainer.save_model("trained_ner_model")
    tokenizer.save_pretrained("trained_ner_model")