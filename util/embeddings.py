from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
sentences = [
    "武当",
    "武林高手们来到了武当",
    "武当",
    "武当是江湖第一大门派",
    "武当",
    "我乃武当弟子李四",
]
embeddings = model.encode(sentences)
print(
    util.cos_sim(
        embeddings[0] + 0.1 * embeddings[1], embeddings[4] + 0.1 * embeddings[5]
    ),
    util.cos_sim(
        embeddings[2] + 0.1 * embeddings[3], embeddings[4] + 0.1 * embeddings[5]
    ),
)
