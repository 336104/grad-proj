import copy
import os
from pathlib import Path
from transformers import AutoTokenizer


# 人物,组织,地点,功夫,武器
class NERBertConfig:
    """model config class"""

    def __init__(
        self,
        data_dir: str = "data/wuxia",
        data_cache_dir: str = "cache/wuxia",
        names_file: str = "data/entities/人物.txt",
        location_file: str = "data/entities/地点.txt",
        orgnization_file: str = "data/entities/组织.txt",
        kongfu_file: str = "data/entities/功夫.txt",
        equipment_file: str = "data/entities/武器.txt",
        checkpoint: str = "google-bert/bert-base-chinese",
        lr: float = 2e-5,
        output_dir: str = "cache/NERBorder",
        train_batch_size: int = 16,
        eval_batch_size: int = 16,
        num_train_epochs: int = 20,
    ):
        self.data_dir = Path(data_dir)
        self.data_cache_dir = Path(data_cache_dir)
        self.names_file = Path(names_file)
        self.location_file = Path(location_file)
        self.orgnization_file = Path(orgnization_file)
        self.kongfu_file = Path(kongfu_file)
        self.equipment_file = Path(equipment_file)
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.lr = lr
        self.output_dir = output_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        if not os.path.exists("cache"):
            os.mkdir("cache")
        if not os.path.exists("log"):
            os.mkdir("log")

    def with_checkpoint(self, checkpoint):
        new_config = copy.deepcopy(self)
        new_config.checkpoint = checkpoint
        new_config.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return new_config


def from_json_config(config):
    return NERBertConfig(**config)
