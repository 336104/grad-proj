from models.border_model import BorderModel
from conf import model_config
model_config.checkpoint='cache/NERBorder/checkpoint-8320'
model = BorderModel(model_config)
print(model.trainer.predict(model.dataset['test']))
