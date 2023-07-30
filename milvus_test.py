from pymilvus import connections, utility, db, CollectionSchema, FieldSchema, DataType, Collection


import random
data = [
  [i for i in range(2000)],
  [str(i) for i in range(2000)],
  [i for i in range(10000, 12000)],
  [[random.random() for _ in range(2)] for _ in range(2000)]
]

  # Milvus Lite has already started, use default_server here.
connections.connect(host='localhost', port=19530)
print(utility.get_server_version())
db.create_database("books")
db.using_database("books")
print("Using database books")
print(db.list_database())
utility.drop_collection('book')

book_id = FieldSchema(
  name="book_id",
  dtype=DataType.INT64,
  is_primary=True,
  )
book_name = FieldSchema(
  name="book_name",
  dtype=DataType.VARCHAR,
  max_length=200,
   )
word_count = FieldSchema(
  name="word_count",
  dtype=DataType.INT64,
   )
book_intro = FieldSchema(
  name="book_intro",
  dtype=DataType.FLOAT_VECTOR,
  dim=2
   )
schema = CollectionSchema(
  fields=[book_id, book_name, word_count, book_intro],
  description="Test book search",
  enable_dynamic_field=True
   )
collection_name = "book"
collection = Collection(
  name=collection_name,
  schema=schema,
  using='default',
  shards_num=2
  )
  
collection = Collection("book")      # Get an existing collection.
mr = collection.insert(data)
collection.flush()

index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
  }

collection = Collection("book")      
collection.create_index(
   field_name="book_intro", 
   index_params=index_params
  )

utility.index_building_progress("book")
collection = Collection("book")      # Get an existing collection.
collection.load()

search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5}
results = collection.search(
data=[[0.1, 0.2]], 
anns_field="book_intro", 
param=search_params,
limit=10, 
expr=None,
	# set the names of the fields you want to retrieve from the search result.
	output_fields=['book_name'],
	consistency_level="Strong"
)

  # get the IDs of all returned hits
results[0].ids

  # get the distances to the query vector from all returned hits
results[0].distances

  # get the value of an output field specified in the search request.
hit = results[0][0]
print(hit.entity)
print(collection.num_entities)