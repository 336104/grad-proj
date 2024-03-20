from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer


# 人物,组织,地点,功夫,武器
class NERBertConfig:
    """model config class"""

    def __init__(
        self,
        data_dir: str = "data/wuxia",
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
        epochs: int = 20,
    ):
        """init model config

        Args:
            data_dir (str, optional): data to train and test model. Defaults to "data/wuxia".
            names_file (str, optional):external name data. Defaults to "data/entities/人物.txt".
            location_file (str, optional):external location data. Defaults to "data/entities/地点.txt".
            orgnization_file (str, optional):external orgnization data. Defaults to "data/entities/组织.txt".
            kongfu_file (str, optional):external kongfu data. Defaults to "data/entities/功夫.txt".
            equipment_file (str, optional):external equipment data. Defaults to "data/entities/武器.txt".
            checkpoint (str, optional): checkpoint of tokenizer and backbone model. Defaults to "google-bert/bert-base-chinese".
            lr (float, optional): learning rate. Defaults to 2e-5.
            output_dir (str, optional): dir to save model. Defaults to "cache/NERBorder".
            train_batch_size (int, optional): train batch size. Defaults to 16.
            eval_batch_size (int, optional): eval batch size. Defaults to 16.
            epochs (int, optional): num of epoch. Defaults to 20.
        """
        self._data_dir = Path(data_dir)
        self._names_file = Path(names_file)
        self._location_file = Path(location_file)
        self._orgnization_file = Path(orgnization_file)
        self._kongfu_file = Path(kongfu_file)
        self._equipment_file = Path(equipment_file)
        self._checkpoint = checkpoint
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._lr = lr
        self._output_dir = output_dir
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self.epochs = epochs

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: str):
        self._data_dir = Path(value)

    @property
    def names_file(self) -> Path:
        return self._names_file

    @names_file.setter
    def names_file(self, value: str):
        self._names_file = Path(value)

    @property
    def location_file(self) -> Path:
        return self._location_file

    @location_file.setter
    def location_file(self, value: str):
        self._location_file = Path(value)

    @property
    def orgnization_file(self) -> Path:
        return self._orgnization_file

    @orgnization_file.setter
    def orgnization_file(self, value: str):
        self._orgnization_file = Path(value)

    @property
    def kongfu_file(self) -> Path:
        return self._kongfu_file

    @kongfu_file.setter
    def kongfu_file(self, value: str):
        self._kongfu_file = Path(value)

    @property
    def equipment_file(self) -> Path:
        return self._equipment_file

    @equipment_file.setter
    def equipment_file(self, value: str):
        self._equipment_file = Path(value)

    @property
    def checkpoint(self) -> str:
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value: str):
        self._tokenizer = AutoTokenizer.from_pretrained(value)
        self._checkpoint = value

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: str):
        self._tokenizer = AutoTokenizer.from_pretrained(value)

    @property
    def lr(self) -> Path:
        return self._lr

    @lr.setter
    def lr(self, value: str):
        self._lr = value

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str):
        self._output_dir = value

    @property
    def train_batch_size(self) -> Path:
        return self._train_batch_size

    @train_batch_size.setter
    def train_batch_size(self, value: str):
        self._train_batch_size = value

    @property
    def eval_batch_size(self) -> Path:
        return self._eval_batch_size

    @eval_batch_size.setter
    def eval_batch_size(self, value: str):
        self._eval_batch_size = value

    @property
    def epochs(self) -> Path:
        return self._epochs

    @epochs.setter
    def epochs(self, value: str):
        self._epochs = value

    def __str__(self) -> str:
        return f"""\ndata_dir : {self.data_dir.as_posix()}\nnames_file : {self.names_file.as_posix()}\nlocation_file : {self.location_file.as_posix()}\norgnization_file : {self.orgnization_file.as_posix()}\nkongfu_file : {self.kongfu_file.as_posix()}\nequipment_file : {self.equipment_file.as_posix()}\ncheckpoint : {self.checkpoint}\nlr : {self.lr}\noutput_dir : {self.output_dir}\ntrain_batch_size : {self.train_batch_size}\neval_batch_size : {self.eval_batch_size}\nepochs : {self.epochs}"""
