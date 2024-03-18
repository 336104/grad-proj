import json
from pathlib import Path
from typing import List, Set, Tuple
from conf import NERBertConfig
import re
from datasets import Dataset
import numpy as np


def gather_all_data(data_dir: Path = NERBertConfig.data_dir) -> List[dict]:
    all_data = []
    for path in data_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data["RecoResult"])
    return all_data


def preprocess_data(data: List[dict]) -> Tuple[List[dict], Set[str]]:
    results = []
    entity_types = set()

    for record in data:
        sentences = re.split(r"[。？！\n]", record["sent_body"])
        for sentence in sentences:
            if sentence == "":
                continue
            tokens = list(sentence)

            entities = []
            for e in record["entities"]:
                e_type, [e_type_name] = e.values()
                entity_types.add(e_type)
                for match in re.finditer(e_type_name, sentence):
                    start, end = match.span()
                    entities.append(
                        {
                            "type": e_type,
                            "type_name": e_type_name,
                            "start": start,
                            "end": end,
                        }
                    )
            entities.sort(key=lambda e: e["start"])
            results.append({"tokens": tokens, "entities": entities})
    return results, entity_types


def add_label(examples):
    all_labels = []
    for tokens, entities in zip(examples["tokens"], examples["entities"]):
        labels = np.ones(len(tokens) + 2, dtype=np.int8)
        labels[0] = -100
        labels[-1] = -100
        flag = 0
        for entity in entities:
            labels[entity["start"] + 1 : entity["end"] + 1] = flag
            flag = 2 - flag
        all_labels.append(labels.tolist())
    return {"labels": all_labels}


def tokenize(examples):
    return NERBertConfig.tokenizer(["".join(tokens) for tokens in examples["tokens"]])


def gen(data):
    for d in data:
        yield d


data = gather_all_data()
results, types = preprocess_data(data)
dataset = Dataset.from_generator(gen, gen_kwargs={"data": results})
dataset = dataset.train_test_split(0.2)
dataset = dataset.map(add_label, batched=True).map(tokenize, batched=True)
