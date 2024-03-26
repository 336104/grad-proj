from util.runner import Runner
from conf import model_config, logger
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.train:
        runner = Runner(model_config)
        runner.train()
    if args.eval:
        eval_config = model_config.with_checkpoint("")
        runner = Runner(eval_config)
        eval_result = runner.eval()
        logger.info(eval_result)
