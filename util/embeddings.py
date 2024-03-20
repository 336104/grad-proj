from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
sentences = ["鸳鸯刀", "张三丰"]
embeddings = model.encode(sentences, convert_to_tensor=True)
print(util.cos_sim(embeddings[0], embeddings[1]))
