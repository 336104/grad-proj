from util.t2s import Converter



def test_converter():
    converter = Converter(Converter.S2T)
    assert converter.converter.convert("你好") == "你好"
