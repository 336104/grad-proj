from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer
from conf import db_config
from util.inference import BorderInferencer
from collections import Counter
from tqdm import trange


def connect():
    connections.connect(**db_config["connection"])


def create_schema():
    entity_id = FieldSchema(name="entity_id", dtype=DataType.INT64, is_primary=True)
    entity = FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=32)
    entity_vector = FieldSchema(
        name="entity_vector", dtype=DataType.FLOAT_VECTOR, dim=db_config["vec_dim"]
    )
    entity_type = FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=32)
    schema = CollectionSchema(fields=[entity_id, entity, entity_vector, entity_type])
    collection = Collection(
        name=db_config["collection_name"],
        schema=schema,
        using=db_config["connection"]["alias"],
    )
    collection.create_index(
        "entity_vector",
        {
            "index_type": db_config["index_type"],
            "metric_type": db_config["metric_type"],
        },
    )
    return collection


def insert_data(collection, dataset):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    all_entities = set()
    for entities in dataset["entities"]:
        for entity in entities:
            all_entities.add((entity["type_name"], entity["type"]))
    type_names = []
    types = []
    for type_name, e_type in all_entities:
        type_names.append(type_name)
        types.append(e_type)
    embeddings = model.encode(type_names)
    data = [
        [i for i in range(len(type_names))],
        type_names,
        embeddings,
        types,
    ]
    collection.insert(data)


def load_to_db(dataset):
    connect()
    if db_config["collection_name"] not in utility.list_collections():
        collection = create_schema()
    else:
        collection = Collection(db_config["collection_name"])
    insert_data(collection, dataset)


def search(collection_name):
    collection = Collection(collection_name)
    collection.load()
    result = collection.search(
        data=[[0.1, 0.2]],
        anns_field="book_intro",
        param={},
        limit=10,
        expr=None,
        output_fields=["book_id"],
        consistency_level="Strong",
    )
    print(result[0].ids)
    print(result[0].distances)
    print(result[0][0].entity.get("book_id"))
    print(result)


def search_entity(collection, embeddings):
    if len(embeddings) == 0:
        return []
    results = collection.search(
        data=embeddings,
        anns_field="entity_vector",
        param={"metric_type": db_config["metric_type"]},
        limit=1,
        output_fields=["entity_type"],
        consistency_level="Strong",
    )
    return [result[0].entity.get("entity_type") for result in results]


def batch_generator(dataset, batch_size):
    for i in trange(len(dataset) // batch_size):
        yield dataset[batch_size * i : batch_size * i + batch_size]


def eval_dataset(dataset, checkpoint):
    connect()
    collection = Collection(db_config["collection_name"])
    collection.load()
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    inferencer = BorderInferencer(checkpoint)
    tp, fn, fp = [1e-6] * 3
    for batch in batch_generator(dataset, 4):
        input_texts = []
        for tokens in batch["tokens"]:
            input_texts.append("".join(tokens))
        predictions = inferencer(input_texts, color_output=False)
        for idx, prediction in enumerate(predictions):
            entities = list(prediction)
            embeddings = model.encode(entities, device="cuda", show_progress_bar=False)
            types = search_entity(collection, embeddings)
            pred_result = Counter(
                [(entity, e_type) for entity, e_type in zip(entities, types)]
            )
            ref_result = Counter(
                [
                    (entity["type_name"], entity["type"])
                    for entity in batch["entities"][idx]
                ]
            )
            tp += len(list((pred_result & ref_result).elements()))
            fn += len(list((ref_result - pred_result).elements()))
            fp += len(list((pred_result - ref_result).elements()))
            precision = tp / (fp + tp)
            recall = tp / (fn + tp)
            f1 = 2 * (precision * recall) / (precision + recall)
            print(precision, recall, f1)
            if f1 < 90:
                print(input_texts[idx], predictions[idx], batch["entities"][idx], types)
                # return
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(precision, recall, f1)
