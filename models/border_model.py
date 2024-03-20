from util.preprocess_data import dataset
from conf import NERBertConfig
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np

data_collator = DataCollatorForTokenClassification(tokenizer=NERBertConfig.tokenizer)


def decode_labels(labels):
    labels.append(-100)
    entities = set()
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            if labels[i - 1] == 0 or labels[i - 1] == 2:
                entities.add((start, i))
            if labels[i] == 0 or labels[i] == 2:
                start = i
    return entities


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    tp, fn, fp = [1e-6] * 3
    for prediction, label in zip(predictions, labels):
        e_pred = decode_labels(prediction.tolist())
        e_ref = decode_labels(label.tolist())
        tp += len(e_pred & e_ref)
        fn += len(e_ref - e_pred)
        fp += len(e_pred - e_ref)
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


model = AutoModelForTokenClassification.from_pretrained(
    NERBertConfig.checkpoint, num_labels=3
)
training_args = TrainingArguments(
    output_dir="mymodel",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=NERBertConfig.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
