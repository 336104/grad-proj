import datetime
import logging
import os


def setup_logger(level=logging.INFO, name="ner"):
    """config logger

    Args:
        level (int, optional): level of logger. Defaults to logging.INFO.
        name (str, optional): name of logger. Defaults to "ner".
    """
    config_root_logger(level)
    config_project_logger(level, name)


def config_root_logger(level=logging.INFO):
    """config root logger

    Args:
        level (int, optional): level of logger. Defaults to logging.INFO.
    """
    logging.basicConfig(
        format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def config_project_logger(level=logging.INFO, name="ner"):
    """config project logger,there is only one project logger

    Args:
        level (int, optional): level of logger. Defaults to logging.INFO.
        name (str, optional): name of logger. Defaults to "ner".
    """
    logger = logging.getLogger(name)
    fmt = logging.Formatter(
        fmt="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not os.path.exists("log"):
        os.mkdir("log")
    fhd = logging.FileHandler(
        "log/ner.log." + str(datetime.datetime.now()), encoding="utf-8"
    )
    fhd.setFormatter(fmt)
    logger.addHandler(fhd)
    logger.setLevel(level)
