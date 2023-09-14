import cProfile
import io
import os
from pstats import SortKey
import pstats
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from typing import List, Any, Tuple
from wannadb.data.data import DocumentBase
from typing import List, Any, Optional, Union
import logging
from wannadb.data.data import DocumentBase, Attribute, Document, InformationNugget
import numpy as np
from wannadb.statistics import Statistics
from wannadb.data.signals import LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, RelativePositionSignal,UserProvidedExamplesSignal, \
    CachedDistanceSignal, CurrentMatchIndexSignal, CombinedEmbeddingSignal
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances


from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTTextEmbedder, SBERTExamplesEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


class vectordb:
    ...
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(vectordb, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the vector database"""
        if hasattr(self, "_already_initialized"):
            return

        self._host = 'localhost'
        self._port = 19530
        self._embedding_identifier = ["TextEmbeddingSignal",
                                      "LabelEmbeddingSignal",
                                      "ContextSentenceEmbeddingSignal"
                                      ]
        self._embedding_collection = None

        logger.info("Vector database initialized")

        self._already_initialized = True

        # Nugget schema
        self._id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
        )
        self._document_id = FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length = 200
        )
        self._embedding_value = FieldSchema( 
            name="embedding_value",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024*3,
        )
        self._embbeding_schema = CollectionSchema(
            fields=[self._id, self._document_id, self._embedding_value],
            description="Schema for nuggets",
            enable_dynamic_field=True,
        )

        # vector index params

        self._index_params = {
        "metric_type":"L2",
        "index_type":"IVF_PQ",
        "params":{"nlist":100, "m":128}
        }

        self._search_params = {
        "metric_type": "L2", 
        "offset": 0, 
        "ignore_growing": False, 
        "params": {"nprobe": 1000}
        }


        
    def __enter__(self) -> connections:
        logger.info("Connecting to vector database")
        connections.connect(alias='default',host=self._host, port=self._port)
        return self 
 
        

    def __exit__(self, *args) -> None:
        logger.info("Disconnecting from vector database")
        connections.disconnect(alias='default')


    def extract_nuggets(self, documentBase: DocumentBase) -> None:
        """
        Extract nugget data from document base
        """
        if "Embeddings" not in utility.list_collections():
             collection = Collection(
                    name="Embeddings",
                    schema=self._embbeding_schema,
                    using="default",
                    shards_num=1,
                )
             logger.info("Created collection Embeddings")

        logger.info("Start extracting nuggets from document base")
        collection = Collection("Embeddings")
        for document in documentBase.documents:
                for id,nugget in enumerate(document.nuggets):

                    combined_embedding = nugget[CombinedEmbeddingSignal]
                    data = [
                        [id],
                        [document.name],
                        [combined_embedding],     
                        ]
                    collection.insert(data)
                    logger.info(f"Inserted nugget {id} {combined_embedding} from document {document.name} into collection")

                collection.flush()
        logger.info("Embedding insertion finished")

        #Vector index
        logger.info("Start indexing")
        collection.create_index(
            field_name='embedding_value', 
            index_params=self._index_params
            )    
        #Scalar index
        collection.create_index(
            field_name='id',
            index_name='scalar_index'
        ) 
        logger.info("Indexing finished")
        logger.info("Extraction finished")

        collection.load()
        self._embedding_collection = Collection('Embeddings')

    def compute_inital_distances(self, attribute_embedding : List[float], document_base: DocumentBase) -> List[Document]:

        remaining_documents: List[Document] = []
        embedding_collection = Collection('Embeddings')

        for i in document_base.documents:
            
            results = embedding_collection.search(
                data=[attribute_embedding], 
                anns_field="embedding_value", 
                param=self._search_params,
                limit=1,
                expr= f"document_id == \"{i.name}\"",
                output_fields=['id'],
                consistency_level="Strong"
            )

            logger.info(f"results for document: {i.name}: Result: {results}")
            logger.info(f"results: {results[0].ids} distance: {results[0].distances} ")
            if results[0].ids: 
                i[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(results[0][0].id) #sicherstellen, dass ; nicht im document name verwendet wird
                i.nuggets[results[0][0].id][CachedDistanceSignal] = CachedDistanceSignal(results[0][0].distance)
                remaining_documents.append(i)
                logger.info(f"Appended nugget: {results[0].ids}; To document {i.name} cached index: {i[CurrentMatchIndexSignal]}; Cached distance {i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal]}")
       
        return remaining_documents
    
    def updating_distances_documents(self, target_embedding: List[float], documents: List[Document]):
        embedding_collection = Collection('Embeddings')

        for i in documents:

            #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
            results = embedding_collection.search(
                data=[target_embedding], 
                anns_field="embedding_value", 
                param=self._search_params,
                limit=1,
                expr= f"document_id == \"{i.name}\"",
                output_fields=['id'],
                consistency_level="Strong"
            )

            if results[0].ids:
                if i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal] > results[0][0].distance:
                    i[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(results[0][0].id)
                    i.nuggets[results[0][0].id][CachedDistanceSignal] = CachedDistanceSignal(results[0][0].distance)

def generate_and_store_embedding(input_path):
    
    with ResourceManager():
        documents = []
        for filename in os.listdir(input_path):
            with open(os.path.join(input_path, filename), "r", encoding='utf-8') as infile:
                text = infile.read()
                documents.append(Document(filename.split(".")[0], text))

        logger.info(f"Loaded {len(documents)} documents")
        death_attribute = Attribute("deaths")
        death_attribute[UserProvidedExamplesSignal] = UserProvidedExamplesSignal(["amount of deaths"])
        date_attribute = Attribute("date")
        date_attribute[UserProvidedExamplesSignal] = UserProvidedExamplesSignal(["Point in time"])
        document_base = DocumentBase(documents, [death_attribute,date_attribute])
        
        # preprocess the data
        default_pipeline = Pipeline([
                            StanzaNERExtractor(),
                            SpacyNERExtractor("SpacyEnCoreWebLg"),
                            ContextSentenceCacher(),
                            CopyNormalizer(),
                            OntoNotesLabelParaphraser(),
                            SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                            SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                            SBERTExamplesEmbedder("SBERTBertLargeNliMeanTokensResource")
            ])
        #BERTContextSentenceEmbedder("BertLargeCasedResource"),

        statistics = Statistics(do_collect=True)
        statistics["preprocessing"]["config"] = default_pipeline.to_config()

        default_pipeline(
                document_base=document_base,
                interaction_callback=EmptyInteractionCallback(),
                status_callback=EmptyStatusCallback(),
                statistics=statistics["preprocessing"]
            )
    
        print(f"Document base has {len([nugget for nugget in document_base.nuggets if 'LabelEmbeddingSignal' in nugget.signals.keys()])} nugget LabelEmbeddings.")
        print(f"Document base has { len([nugget for nugget in document_base.nuggets if 'TextEmbeddingSignal' in nugget.signals.keys()])} nugget TextEmbeddings.")
        print(f"Document base has { len([nugget for nugget in document_base.nuggets if 'ContextSentenceEmbeddingSignal' in nugget.signals.keys()])} nugget ContextSentenceEmbeddings.")
        print(f"Document base has { len([attribute for attribute in document_base.attributes if 'LabelEmbeddingSignal' in attribute.signals.keys()])} attribute LabelEmbeddings.")
        print(f"Document base has { len([attribute for attribute in document_base.attributes if 'TextEmbeddingSignal' in attribute.signals.keys()])} attribute TextEmbeddings.")
        print(f"Document base has { len([attribute for attribute in document_base.attributes if 'ContextSentenceEmbeddingSignal' in attribute.signals.keys()])} attribute ContextSentenceEmbeddings.")
        
        with vectordb() as vb:
            for i in utility.list_collections():
                    utility.drop_collection(i)

            vb.extract_nuggets(document_base)
            
        with open("corona.bson", "wb") as file:
            file.write(document_base.to_bson())


#generate_and_store_embedding('C:\\Users\\Pascal\\Desktop\\WannaDB\\lab23_wannadb_scale-up\\datasets\\corona\\raw-documents')

    

        

