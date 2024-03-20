import opencc
from pathlib import Path
from conf import NERBertConfig
from typing import List
import json


class Converter:
    S2T = "s2t.json"
    T2S = "t2s.json"

    def __init__(self, config):
        self.converter = opencc.OpenCC(config)

    def convert_file(self, path: Path) -> str:
        print(path)
        if path.exists():
            print(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.converter.convert(content)
        else:
            return ""

    def convert_entities(self):
        name = self.convert_file(NERBertConfig.names_file)
        orgnization = self.convert_file(NERBertConfig.orgnization_file)
        location = self.convert_file(NERBertConfig.location_file)
        kongfu = self.convert_file(NERBertConfig.kongfu_file)
        equipment = self.convert_file(NERBertConfig.equipment_file)
        return {
            "name": name,
            "orgnization": orgnization,
            "location": location,
            "kongfu": kongfu,
            "equipment": equipment,
        }

    def gather_wuxia(self, data_dir: Path = NERBertConfig.data_dir) -> List[dict]:
        all_data = []
        for path in data_dir.glob("*.json"):
            content = self.convert_file(path)
            data = json.loads(content)
            all_data.extend(data["RecoResult"])
        return all_data


if __name__ == "__main__":
    converter = Converter(Converter.T2S)
    print(converter.convert_entities()["name"])
