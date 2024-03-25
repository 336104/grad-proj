import opencc
from pathlib import Path
from conf import model_config, logger
from typing import Dict, List
import json


class Converter:

    S2T = "s2t.json"
    T2S = "t2s.json"

    def __init__(self, config):
        """init converter

        Args:
            config (str): choose from 'Converter.S2T' and 'Converter.T2S'

        Raises:
            Exception: not choose right config
        """
        if config == Converter.S2T:
            self.task = "简体转繁体"
        elif config == Converter.T2S:
            self.task = "繁体转简体"
        else:
            logger.exception("config参数请从'Converter.S2T'和'Converter.T2S'选择")
            raise Exception("未选择正确参数")
        self.converter = opencc.OpenCC(config)

    def convert_file(self, path: Path) -> str:
        """convert a file to the language described by self.task

        Args:
            path (Path): path of the file to be converted

        Returns:
            str: read file and convert it to string
        """
        logger.info(path)
        if path.exists():
            logger.info(f"正在转换{path}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.converter.convert(content)
        else:
            logger.error(f"文件{path}不存在")
            return ""

    def convert_entities(self) -> Dict:
        """convert external data

        Returns:
            str: read all files and convert them to dict of string
        """
        logger.info(f"转换外部数据集:{self.task}")
        name = self.convert_file(model_config.names_file)
        orgnization = self.convert_file(model_config.orgnization_file)
        location = self.convert_file(model_config.location_file)
        kongfu = self.convert_file(model_config.kongfu_file)
        equipment = self.convert_file(model_config.equipment_file)
        return {
            "name": name,
            "orgnization": orgnization,
            "location": location,
            "kongfu": kongfu,
            "equipment": equipment,
        }

    def gather_data(self, data_dir: Path = model_config.data_dir) -> List[dict]:
        """gather all data to train and test model

        Args:
            data_dir (Path, optional): path of dataset. Defaults to model_config.data_dir.

        Returns:
            List[dict]: list of dataset
        """
        logger.info(f"转换模型所需数据集:{self.task}")
        all_data = []
        for path in data_dir.glob("*.json"):
            content = self.convert_file(path)
            data = json.loads(content)
            all_data.extend(data["RecoResult"])
        logger.info(f"共有{len(all_data)}条数据")
        return all_data


if __name__ == "__main__":
    converter = Converter(Converter.T2S)
    print(converter.convert_entities()["name"])
