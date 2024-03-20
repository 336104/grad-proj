from transformers import AutoModelForTokenClassification
from conf import NERBertConfig

tokenized_input = NERBertConfig.tokenizer("张三丰拿出一把鸳鸯刀", return_tensors="pt")
model = AutoModelForTokenClassification.from_pretrained("./mymodel/checkpoint-416")
output = model(**tokenized_input)
tag = output.logits.argmax(dim=-1)
print(tag[..., 1:-1])
