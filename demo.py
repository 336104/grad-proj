from util.runner import Runner
from conf import model_config

runner = Runner(model_config)
runner.train()
