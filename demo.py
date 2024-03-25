from util.runner import Runner
from conf import model_config
from transformers import BertForTokenClassification
runner = Runner(model_config)
runner.train()
