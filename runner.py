from models.border_model import BorderModel
from conf import model_config
model = BorderModel(config=model_config)
model.train(push_to_hub=True)
