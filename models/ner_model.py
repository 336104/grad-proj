from dataclasses import dataclass
from data import dataset, tokenizer, id2label
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)
from transformers.utils import PaddingStrategy
from models.bert.bert_injection import BertForTokenClassification_
import copy
import evaluate
import numpy as np
import os
from typing import Optional, Union
from transformers.data.data_collator import DataCollatorMixin

os.environ["CUDA_VISIBLE_DEVICES"] = ""
dataset = copy.deepcopy(dataset)
dataset = dataset.rename_column("type_labels", "labels")


@dataclass
class MyDataCollator(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        border_labels = [feature["border_labels"] for feature in features]
        no_labels_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "border_labels"
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label)
                + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
            batch["border_labels"] = [
                to_list(label)
                + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in border_labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label))
                + to_list(label)
                for label in labels
            ]
            batch["border_labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label))
                + to_list(label)
                for label in border_labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        batch["border_labels"] = torch.tensor(batch["border_labels"], dtype=torch.int64)
        return batch


data_collator = MyDataCollator(tokenizer=tokenizer)
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


model = BertForTokenClassification_.from_pretrained("bert-base-uncased", num_labels=9)


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
    report_to="none",
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
