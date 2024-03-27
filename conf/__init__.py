import logging
import json
from .log_config import setup_logger
from .model_config import NERBertConfig, from_json_config

# 配置日志
project_logger_name = "ner"
project_logging_level = logging.INFO
setup_logger(level=project_logging_level, name=project_logger_name)
logger = logging.getLogger(project_logger_name)

# 配置模型参数
with open("./conf/model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
active = config["active"]
model_config = from_json_config(config[active])
# 数据库参数
with open("./conf/db_config.json", "r", encoding="utf-8") as f:
    db_config = json.load(f)
__all__ = [NERBertConfig, model_config, project_logger_name, logger, db_config]
