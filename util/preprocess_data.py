import logging
from typing import List, Set, Tuple
from conf import model_config
import re
from datasets import Dataset, load_from_disk
import numpy as np
from util.t2s import Converter
from tqdm import tqdm
import os

def preprocess_data(data: List[dict]) -> Tuple[List[dict], Set[str]]:
    results = []
    entity_types = set()

    for record in tqdm(data):
        sentences = re.split(r"[。？！\n]", record["sent_body"])
        for sentence in sentences:
            if sentence == "":
                continue
            tokenized_inputs = model_config.tokenizer(sentence)
            tokens = model_config.tokenizer.convert_ids_to_tokens(
                tokenized_inputs["input_ids"]
            )
            sentence = "".join(tokens)
            strIndex_to_listIndex = []
            for idx, token in enumerate(tokens):
                strIndex_to_listIndex.extend([idx] * len(token))
            entities = []
            for e in record["entities"]:
                e_type, [e_type_name] = e.values()
                if len(e_type_name) < 1:
                    continue
                entity_types.add(e_type)
                for match in re.finditer(e_type_name, sentence):
                    start, end = match.span()
                    entities.append(
                        {
                            "type": e_type,
                            "type_name": e_type_name,
                            "start": strIndex_to_listIndex[start],
                            "end": strIndex_to_listIndex[end],
                        }
                    )
            entities.sort(key=lambda e: e["start"])
            tokenized_inputs.update({"tokens": tokens, "entities": entities})
            results.append(tokenized_inputs)
    return results, entity_types


def add_label(examples):
    all_labels = []
    for tokens, entities in zip(examples["tokens"], examples["entities"]):
        labels = np.ones(len(tokens), dtype=np.int8)
        labels[0] = -100
        labels[-1] = -100
        flag = 0
        for entity in entities:
            labels[entity["start"] : entity["end"]] = flag
            flag = 2 - flag
        all_labels.append(labels.tolist())
    return {"labels": all_labels}


def gen(data):
    for d in data:
        yield d


def load_data(regenerate=False):
    if regenerate or not os.path.exists('cache/wuxia'):
        converter = Converter(Converter.T2S)
        data = converter.gather_wuxia()
        results, types = preprocess_data(data)
        logging.info("start")
        dataset = Dataset.from_generator(gen, gen_kwargs={"data": results})
        logging.info("end")
        dataset = dataset.train_test_split(0.2, shuffle=True)
        dataset = dataset.map(add_label, batched=True)
        dataset.save_to_disk("cache/wuxia")
        return dataset
    else:
        dataset = load_from_disk("cache/wuxia")
        return dataset
