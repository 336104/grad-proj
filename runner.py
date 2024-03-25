from models.border_model import BorderModel
from conf import model_config

# model_config.checkpoint = "H336104/NERBorder"
model_config.checkpoint = "./cache/NERBorder/checkpoint-4576"
model = BorderModel(config=model_config)
# model.train()
print(model.predict(model.dataset["test"]))
