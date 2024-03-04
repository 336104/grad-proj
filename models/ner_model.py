from data import dataset, tokenizer, id2label
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from models.bert.bert_injection import BertForTokenClassification_
import copy
import evaluate
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
dataset = copy.deepcopy(dataset)
dataset = dataset.rename_column("type_labels", "labels")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metrics = evaluate.load("seqeval")


def compute_metrics(p):
    batch_predictions, batch_labels = p
    batch_predictions = np.argmax(batch_predictions, axis=-1)

    for predictions, labels in zip(batch_predictions, batch_labels):
        valid_predictions = []
        valid_labels = []
        for prediction, label in zip(predictions, labels):
            if label != -100:
                valid_predictions.append(id2label[prediction])
                valid_labels.append(id2label[label])
        metrics.add(prediction=valid_predictions, reference=valid_labels)
    return metrics.compute()


model = BertForTokenClassification_.from_pretrained("bert-base-uncased", num_labels=7)


training_args = TrainingArguments(
    output_dir="mymodel",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
