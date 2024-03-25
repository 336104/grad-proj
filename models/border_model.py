from util.preprocess_data import load_data
from conf import model_config, NERBertConfig
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from metrics import BorderMetric


class BorderModel:
    def __init__(self, config: NERBertConfig):
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=model_config.tokenizer
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_config.checkpoint, num_labels=3
        )
        self.dataset = load_data()
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            learning_rate=config.lr,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            num_train_epochs=config.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=4,
            logging_steps=100,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=model_config.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=BorderMetric.compute_metrics,
        )

    def train(self, push_to_hub: bool = False):
        self.trainer.train()
        if push_to_hub:
            self.trainer.push_to_hub()

    def eval(self):
        self.trainer.evaluate()

    def predict(self):
        return self.trainer.predict()
