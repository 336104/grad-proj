from util.runner import Runner
from util.milvus_util import load_to_db, eval_dataset
from util.preprocess_data import load_data
from conf import model_config, logger
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bert", action="store_true")
    parser.add_argument("--eval_bert", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--load_to_db", action="store_true")
    parser.add_argument("--eval_search", action="store_true")
    args = parser.parse_args()

    if args.train_bert:
        runner = Runner(model_config)
        runner.train()
    if args.eval_bert:
        eval_config = model_config.with_checkpoint("cache/NERBorder/" + args.checkpoint)
        runner = Runner(eval_config)
        eval_result = runner.eval()
        logger.info(eval_result)
    if args.load_to_db:
        dataset = load_data(model_config)
        load_to_db(dataset["train"])
    if args.eval_search:
        dataset = load_data(model_config)
        eval_dataset(dataset["test"], "cache/NERBorder/" + args.checkpoint)
