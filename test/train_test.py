from models.border_model import BorderModel
from conf import model_config


def test_train():
    model_config.epochs = 1
    model = BorderModel(config=model_config)
    model.train(push_to_hub=True)
