from data import dataset, tokenizer
from transformers import DataCollatorForTokenClassification
import copy

dataset = copy.deepcopy(dataset)
dataset = dataset.rename_column("type_labels", "labels")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
data_collator(dataset["train"])
