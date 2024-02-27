from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from pathlib import Path
from conf import BaseConfig

cache_dir = Path(__file__).parent / "cache" / BaseConfig.dataset.replace("/", "-")
print(cache_dir.absolute().as_posix())
if not cache_dir.exists():
    dataset = load_dataset(BaseConfig.dataset)
    dataset.save_to_disk(cache_dir)
else:
    dataset = load_from_disk(cache_dir)
id2label = dataset["train"].features["ner_tags"].feature.names
id2label = {k: v for k, v in enumerate(id2label)}
label2id = {v: k for k, v in id2label.items()}
tokenizer = AutoTokenizer.from_pretrained(BaseConfig.base_model)


def add_border_labels(examples):
    border_labels = []
    for tags in examples["ner_tags"]:
        border_label = []
        border_flag = 1
        for tag in tags:
            if tag == 0:
                border_label.append(1)
            elif tag % 2 == 1:
                border_flag = -border_flag
                border_label.append(border_flag + 1)
            else:
                border_label.append(border_flag + 1)
        border_labels.append(border_label)
    return {"border_labels": border_labels}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["border_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = dataset.map(add_border_labels, batched=True).map(
    tokenize_and_align_labels, batched=True
)
