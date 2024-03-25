from util.preprocess_data import load_data

dataset = load_data(regenerate=True)
print(dataset["train"][0])
