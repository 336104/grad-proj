from conf import NERBertConfig
from util.preprocess_data import dataset

flag = False
for sample in dataset["train"]:
    for tokens in sample["tokens"]:
        # print(NERBertConfig.tokenizer.tokenize(tokens))
        # break
        tok = NERBertConfig.tokenizer.tokenize(tokens)
        if tok == []:
            flag = True
            print(tokens)
            break
    if flag:
        print(sample)
        break
