import logging
from .log_config import setup_logger
from .model_config import NERBertConfig

# 配置日志
project_logger_name = "ner"
project_logging_level = logging.INFO
# setup_logger(level=project_logging_level, name=project_logger_name)
logger = logging.getLogger(project_logger_name)

# 配置模型参数
model_config = NERBertConfig()
__all__ = [NERBertConfig, model_config, project_logger_name, logger]
