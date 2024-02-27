from data import dataset, tokenizer
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metrics = evaluate.combine(["f1", "precision", "recall"])


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    for prediction, label in zip(predictions, labels):
        for p, l in zip(prediction, label):
            if l != -100:
                metrics.add(p, l)
    return metrics.compute(average="macro")


model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)
training_args = TrainingArguments(
    output_dir="mymodel",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    do_train=False,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
    logging_strategy="steps",
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