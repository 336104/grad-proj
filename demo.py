from util.preprocess_data import dataset

dataset = dataset.remove_columns(["tokens", "entities"])
for sample in dataset["test"]:
    if len(sample["input_ids"]) != len(sample["labels"]):
        print(sample)
    break
