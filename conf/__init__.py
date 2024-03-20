from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer


# 人物,组织,地点,功夫,武器
class NERBertConfig:
    data_dir: Path = Path("data/wuxia")
    names_file: Path = Path("data/entities/人物.txt")
    location_file: Path = Path("data/entities/地点.txt")
    orgnization_file: Path = Path("data/entities/组织.txt")
    kongfu_file: Path = Path("data/entities/功夫.txt")
    equipment_file: Path = Path("data/entities/武器.txt")
    checkpoint: str = "google-bert/bert-base-chinese"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
