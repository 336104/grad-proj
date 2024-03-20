from transformers import AutoModelForTokenClassification, AutoTokenizer
from colorama import Back, Style
from collections import deque


class BorderInferencer:
    def __init__(self, checkpoint) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint)

    def decode_labels(self, labels):
        labels.append(-100)
        entities = []
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                if labels[i - 1] == 0 or labels[i - 1] == 2:
                    if i - start != 1:
                        entities.append((start, i))
                if labels[i] == 0 or labels[i] == 2:
                    start = i
        return entities

    def decode(self, tokenized_input, labels):
        all_entities = []
        for i in range(labels.size(0)):

            input_ids = tokenized_input["input_ids"][i]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            tags = labels[i][tokenized_input["attention_mask"][i].bool()][1:-1].tolist()
            tokens = tokens[1 : len(tags) + 1]
            entities = self.decode_labels(tags)
            sample_entities = deque()
            for start, end in reversed(entities):
                sample_entities.appendleft("".join(tokens[start:end]))
                tokens.insert(end, Style.RESET_ALL)
                tokens.insert(start, Back.YELLOW)
            print("".join(tokens))
            all_entities.append(sample_entities)
        return all_entities

    def __call__(self, text):
        tokenized_input = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        output = self.model(**tokenized_input)
        labels = output.logits.argmax(dim=-1)
        return self.decode(tokenized_input, labels)
        # return [
        #     tag[i][tokenized_input["attention_mask"][i].bool()][1:-1].tolist()
        #     for i in range(tag.size(0))
        # ]


if __name__ == "__main__":
    cls = BorderInferencer("mymodel/checkpoint-416")
    print(
        cls(
            [
                "123明朝万历四十三年，武当派弟子卓一航，不顾师门恩怨，与闻名江湖的女侠练霓裳相爱情深，因此遭受同门无情的诬陷与折磨，两人历经生死，不但武艺大进而且情愫更增。明神宗驾崩，光宗继位，但遭遇奸佞宦官魏忠贤所害，另立年幼的熹宗继位，魏逆掌持朝政，独揽大权，并与女真可汗努尔哈赤勾结，欲颠覆明室，自立为皇。\n慕容冲是练霓裳的师兄，实为女真密使，他为了一统武林，不惜协助女真人入侵中原，并用计使练霓裳与卓一航反目，致使练霓裳一夜之间青丝尽白，愤而出走塞外，痴情的卓一航远赴塞外寻找，途中又遭到慕容冲的谋害。皇天不负有心人，卓一航与练霓裳这对苦命鸳鸯终得相见，冰释误会。两人遂召集江湖英雄，在山海关击破女真大军，使皇太极一时无法入侵中原。\n熹宗驾崩之后，魏忠贤被崇祯皇帝赐死。崇祯受奸臣谗言，杀害抗清大将袁崇焕，致使明朝江山从此瓦解。卓一航与练霓裳对国事失望至极，双双退隐江湖，永享神仙眷侣的生活。",
                "航是官家子弟，自幼被送上武当习文武；上山之日恰好遇到天都，原来天都已向武当下战书，要夺武当的珍藏剑谱，就在天都获胜之际，都的师妹华竟偷去剑谱，华山众人以为两人合谋，向两人大打出手。不幸双方都深受重伤，这时一航出现，航父命一航杀掉都和华，航不愿杀人。都先行恢复，原想杀掉武当弟子，航求情，都放弃杀人，留下剑谱带走受重伤的师妹离去。武当因而重获剑谱。武当掌门念航心慈仁厚，破例收一航为徒弟。经此一役，都接任天都掌门之位，并坚守不准嫁娶的门规，却原来华早锺情都，华含悢留书归隐，约定廿年后再比武，都此时已追悔莫及，静待廿年后与华重聚。",
            ]
        )
    )
