import os
from util.t2s import Converter


def setup_module():
    print("setup_function--->")


def test_converter():
    converter = Converter(Converter.S2T)
    assert converter.converter.convert("你好") == "你好"


def test_import():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
