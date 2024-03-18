from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer


class NERBertConfig:
    data_dir: Path = Path("data/wuxia")
    checkpoint: str = "google-bert/bert-base-chinese"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
