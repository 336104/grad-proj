from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import random


def connect():
    connections.connect(alias="default", host="localhost", port="19530")


def prepare_schema():
    book_id = FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True)
    book_name = FieldSchema(
        name="book_name",
        dtype=DataType.VARCHAR,
        max_length=200,
        default_value="Unknown",
    )
    word_count = FieldSchema(
        name="word_count", dtype=DataType.INT64, default_value=9999
    )
    book_intro = FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=2)
    schema = CollectionSchema(
        fields=[book_id, book_name, word_count, book_intro],
        description="Test book search",
        enable_dynamic_field=True,
    )
    collection_name = "book"

    return schema, collection_name


def create_schema(schema, collection_name):
    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )
    return collection


def insert_data(collection_name):
    collection = Collection(collection_name)
    data = [
        [i for i in range(2000)],
        [str(i) for i in range(2000)],
        [i for i in range(10000, 12000)],
        [[random.random() for _ in range(2)] for _ in range(2000)],
    ]
    collection.insert(data)


def create_index(collection_name):
    collection = Collection(collection_name)
    collection.create_index("book_intro", {"index_type": "FLAT", "metric_type": "L2"})


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


if __name__ == "__main__":
    connect()
    # create_index("book")
    search("book")
