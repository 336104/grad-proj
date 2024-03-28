from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
sentences = [
    "明朝万历四十三年，武当派弟子卓一航，不顾师门恩怨，与闻名江湖的女侠练霓裳相爱情深",
    "明朝万历四十三年，武当派弟子[MASK]，不顾师门恩怨，与闻名江湖的女侠练霓裳相爱情深",
    "痴情的卓一航远赴塞外寻找，途中又遭到慕容冲的谋害",
    "痴情的[MASK]远赴塞外寻找，途中又遭到慕容冲的谋害",
]

embedding = model.encode(sentences, output_value="token_embeddings")

# print(util.cos_sim((embedding[0] - embedding[1]), (embedding[2] - embedding[3])))
print("hi")
