class BaseConfig:
    dataset: str = "conll2003"
    base_model: str = "bert-base-uncased"
    ent_types: int = 4