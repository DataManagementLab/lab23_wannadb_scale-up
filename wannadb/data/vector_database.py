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

from typing import List, Any
from wannadb.data.data import DocumentBase
from typing import List, Any, Optional, Union
import re
import logging
from wannadb.data.data import DocumentBase, InformationNugget, Attribute, Document
import random
import time
import numpy as np
from wannadb.statistics import Statistics
from wannadb.data.signals import LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, RelativePositionSignal, POSTagsSignal, CurrentMatchIndexSignal, CachedDistanceSignal
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances


from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import BERTContextSentenceEmbedder, RelativePositionEmbedder, SBERTTextEmbedder, SBERTLabelEmbedder, FastTextLabelEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback

# Nugget schema
id = FieldSchema(
    name="id",
    dtype=DataType.VARCHAR,
    max_length=200,
    is_primary=True,
)
embedding_type = FieldSchema( # embedding type
    name="embedding_type",
    dtype=DataType.VARCHAR,
    max_length=200,
)
embedding_value = FieldSchema( # embedding value
    name="embedding_value",
    dtype=DataType.FLOAT_VECTOR,
    dim=2048,
)
embbeding_schema = CollectionSchema(
    fields=[id, embedding_value],
    description="Schema for nuggets",
    enable_dynamic_field=True,
)

# vector index params

INDEX_PARAMS = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}


logger: logging.Logger = logging.getLogger(__name__)

VECTORDB: Optional["vectordb"] = None

class vectordb:
    """
    Interface for vector database.
    """

    def __init__(self) -> None:
        """Initialize the vector database"""
        global VECTORDB

        if VECTORDB is not None:
            logger.error("Vector database already initialized")
            assert False, "There can only be one vector database"
        else:
            VECTORDB = self

        self._host = 'localhost'
        self._port = 19530
        self._embedding_identifier = [
                "LabelEmbeddingSignal",
                "TextEmbeddingSignal"
            ]

        logger.info("Vector database initialized")
        
        
    

    def __enter__(self) -> connections:
        logger.info("Connecting to vector database")
        connections.connect(alias='default',host=self._host, port=self._port)
        return self 
 
        

    def __exit__(self, *args) -> None:
        global VECTORDB
        logger.info("Disconnecting from vector database")
        connections.disconnect(alias='default')
        VECTORDB = None
        
        
    def regenerate_index(self, index_params, collection_name = "Embeddings"):
        collection = Collection(collection_name)
        try:
            collection.release()
        except Exception:
            pass
        
        collection.drop_index()
        collection.create_index(
            field_name='embedding_value', 
            index_params=index_params
            )
        


    def extract_nuggets(self, documentBase: DocumentBase, index_params=INDEX_PARAMS, collection_name = "Embeddings") -> None:
        """
        Extract nugget data from document base
        """
        if collection_name not in utility.list_collections():
             collection = Collection(
                    name=collection_name,
                    schema=embbeding_schema,
                    using="default",
                    shards_num=1,
                )
             logger.info("Created collection Embeddings")

        logger.info("Start extracting nuggets from document base")
        collection = Collection(collection_name)
        for document in documentBase.documents:
                for id,nugget in enumerate(document.nuggets):
                    embedding_vector = []
                    for embedding_name in self._embedding_identifier:
                        embedding_vector_data = nugget.signals.get(embedding_name, None)
                        if embedding_vector_data is not None:
                            embedding_value = embedding_vector_data.value
                        else:
                            embedding_value = np.zeros(1024)    
                            
                        embedding_vector.extend(embedding_value)
                    
                    embedding_vector = np.array(embedding_vector)
                    data = [
                        [f"{document.name};{str(nugget._start_char)};{str(nugget._end_char)}"],
                        [embedding_vector],     
                    ]
                    collection.insert(data)
                    logger.info(f"Inserted nugget {id} from document {document.name} into collection {document.name}")
                collection.flush()
        logger.info("Embedding insertion finished")
        logger.info("Start indexing")
        collection.create_index(
            field_name='embedding_value', 
            index_params=index_params
            )         
        logger.info("Indexing finished")
        logger.info("Extraction finished")
    
    
    def compute_inital_distances(self, attribute_embedding : List[float], document_base: DocumentBase) -> List[Document]:

        remaining_documents: List[Document] = []
        embedding_collection = Collection('Embeddings')
        embedding_collection.load()
        

        for i in document_base.documents:

            #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
            search_param= {
                'data' : [attribute_embedding],
                'anns_field' : "embedding_value",
                'param' : {"metric_type": "L2", "params": {"nprobe": 1000}, "offset": 0},
                'limit' : 1,
                'expr' : f"id like \"{i.name}%\""
                    }
            
            results = embedding_collection.search(**search_param)
            logger.info(f"results for document: {i.name}: Result: {results}")
            logger.info(f"results: {results[0].ids} distance: {results[0].distances} ")
            if results[0].ids: 
                i[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(int(results[0][0].id.split(";")[1])) #sicherstellen, dass ; nicht im document name verwendet wird
                i.nuggets[int(results[0][0].id.split(";")[1])][CachedDistanceSignal] = CachedDistanceSignal(results[0][0].distance)
                remaining_documents.append(i)
                logger.info(f"Appended nugget: {results[0].ids}; To document {i.name} cached index: {i[CurrentMatchIndexSignal]}; Cached distance {i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal]}")
       
        embedding_collection.release()
        return remaining_documents

def compute_distances(
            xs,
            ys
    ) -> np.ndarray:

        assert len(xs) > 0 and len(ys) > 0, "Cannot compute distances for an empty collection!"
        if not isinstance(xs, list):
            xs = list(xs)
        if not isinstance(ys, list):
            ys = list(xs)

        signal_identifiers: list[str] = [
            LabelEmbeddingSignal.identifier,
            TextEmbeddingSignal.identifier,
            ContextSentenceEmbeddingSignal.identifier,
            RelativePositionSignal.identifier,
            POSTagsSignal.identifier
        ]

        # check that all xs and all ys contain the same signals
        xs_is_present: np.ndarray = np.zeros(5)
        for idx in range(5):
            if signal_identifiers[idx] in xs[0].signals.keys():
                xs_is_present[idx] = 1
        for x in xs:
            for idx in range(5):
                    if (
                            xs_is_present[idx] == 1
                            and signal_identifiers[idx] not in x.signals.keys()
                            or xs_is_present[idx] == 0
                            and signal_identifiers[idx] in x.signals.keys()
                    ):
                        assert False, "All xs must have the same signals!"

        ys_is_present: np.ndarray = np.zeros(5)
        for idx in range(5):
            if signal_identifiers[idx] in ys[0].signals.keys():
                ys_is_present[idx] = 1
        for y in ys:
            for idx in range(5):
                    if (
                            ys_is_present[idx] == 1
                            and signal_identifiers[idx] not in y.signals.keys()
                            or ys_is_present[idx] == 0
                            and signal_identifiers[idx] in y.signals.keys()
                    ):
                        assert False, "All ys must have the same signals!"

        # compute distances signal by signal
        start_time = time.time()
        amount_distances=0
        distances: np.ndarray = np.zeros((len(xs), len(ys)))
        for idx in range(3):
            if xs_is_present[idx] == 1 and ys_is_present[idx] == 1:
                x_embeddings: np.ndarray = np.array([x[signal_identifiers[idx]] for x in xs])
                y_embeddings: np.ndarray = np.array([y[signal_identifiers[idx]] for y in ys])
                tmp: np.ndarray = cosine_distances(x_embeddings, y_embeddings)
                distances = np.add(distances, tmp)
                amount_distances += len(tmp)

        if xs_is_present[3] == 1 and ys_is_present[3] == 1:
            x_positions: np.ndarray = np.array([x[signal_identifiers[3]] for x in xs])
            y_positions: np.ndarray = np.array([y[signal_identifiers[3]] for y in ys])
            tmp: np.ndarray = np.zeros((len(x_positions), len(y_positions)))
            for x_ix, x_value in enumerate(x_positions):
                for y_ix, y_value in enumerate(y_positions):
                    tmp[x_ix, y_ix] = np.abs(x_value - y_value)
            distances = np.add(distances, tmp)
            amount_distances += len(tmp)

        if xs_is_present[4] == 1 and ys_is_present[4] == 1:
            x_values: list[list[str]] = [x[signal_identifiers[4]] for x in xs]
            y_values: list[list[str]] = [y[signal_identifiers[4]] for y in ys]
            tmp: np.ndarray = np.ones((len(x_values), len(y_values)))
            for x_ix, x_value in enumerate(x_values):
                for y_ix, y_value in enumerate(y_values):
                    if x_value == y_value:
                        tmp[x_ix, y_ix] = 0
            distances = np.add(distances, tmp)
            amount_distances += len(tmp)

        print(f"Processed distances without VDB: {amount_distances}")
        actually_present: np.ndarray = xs_is_present * ys_is_present
        if np.sum(actually_present) == 0:
            print("Without VDB--- %s seconds ---" % (time.time() - start_time))
            return np.ones_like(distances)
        else:
            print("Without VDB:--- %s seconds ---" % (time.time() - start_time))
            return np.divide(distances, np.sum(actually_present))


def generate_and_store_embedding(input_path, index_params = INDEX_PARAMS):
    
    with ResourceManager() as resource_manager:
        documents = []
        for filename in os.listdir(input_path):
            with open(os.path.join(input_path, filename), "r", encoding='utf-8') as infile:
                text = infile.read()
                documents.append(Document(filename.split(".")[0], text))

        logger.info(f"Loaded {len(documents)} documents")
        document_base = DocumentBase(documents, [Attribute("deaths"),Attribute("date")])
        
        # preprocess the data
        default_pipeline = Pipeline([
                            StanzaNERExtractor(),
                            SpacyNERExtractor("SpacyEnCoreWebLg"),
                            ContextSentenceCacher(),
                            CopyNormalizer(),
                            OntoNotesLabelParaphraser(),
                            SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                            SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
                            SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
            ])
        StanzaNERExtractor(),
        SpacyNERExtractor("SpacyEnCoreWebLg"),
        ContextSentenceCacher(),
        CopyNormalizer(),
        OntoNotesLabelParaphraser(),
        SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
        SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
        SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
        BERTContextSentenceEmbedder("BertLargeCasedResource"),
        RelativePositionEmbedder()
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

            start_time_ = time.time()
            vb.extract_nuggets(document_base, index_params)
            print("Extraction: " + str(time.time() - start_time_))
            
        with open("corona.bson", "wb") as file:
            file.write(document_base.to_bson())


def generate_new_index(index_params):
    with vectordb() as vb:
        vb.regenerate_index(index_params)
        
        
def compute_new_vdb_distances(path = "corona.bson"):
    
    times = {}
    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())
    
        with vectordb() as vb:
            for attribute in document_base.attributes:
                #for every embedding signal in the attribute
                #for i in [signal.identifier for signal in attribute.signals.values() if signal.identifier in embeddding_signals]:

                embedding_vector = []
                for embedding_name in vb._embedding_identifier:
                    embedding_vector_data = attribute.signals.get(embedding_name, None)
                    if embedding_vector_data is not None:
                        embedding_value = embedding_vector_data.value
                        
                    else:
                        embedding_value = np.zeros(1024)
                    embedding_vector.extend(embedding_value)
                
                start_time = time.time()
                vb.compute_inital_distances(embedding_vector, document_base)
                times[attribute.name] = time.time() - start_time
                
    return times
                

def compute_embedding_distances(path = "corona.bson", rounds= 1, nprobe_max= 100, max_limit=2001 ):    
    
    # pr = cProfile.Profile()
    
    
    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            logger.error("Document base is inconsistent!")
            return
                
        with vectordb() as vb:
            # pr.enable()
            
            output = {}
            start_time_ = time.time()
            embedding_collection = Collection("Embeddings")
            embedding_collection.load()
            print("Load Collection:--- %s seconds ---" % (time.time() - start_time_))
            print(f"Embedding collection has {embedding_collection.num_entities} embeddings.")
            #dist_collection = Collection("Distances")
            #dist_collection.load()
            distances={}
            
            for nprobe in range(20,nprobe_max, 20):
                search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}, "offset": 0}
                output[nprobe] = {}
                for limit in range(500, max_limit, 500):
                    time_sum = 0
                    amount_distances = 0
                    for _ in range(rounds):
                        
                        #for every attribute in the document base
                        start_time = time.time()
                        for attribute in document_base.attributes:
                            distances[attribute.name]={}
                            #for every embedding signal in the attribute
                            #for i in [signal.identifier for signal in attribute.signals.values() if signal.identifier in embeddding_signals]:

                            embedding_vector = []
                            for embedding_name in vb._embedding_identifier:
                                embedding_vector_data = attribute.signals.get(embedding_name, None)
                                if embedding_vector_data is not None:
                                    embedding_value = embedding_vector_data.value
                                    
                                else:
                                    embedding_value = np.zeros(1024)
                                embedding_vector.extend(embedding_value)

                            #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
                            results = embedding_collection.search(
                                data = [embedding_vector],
                                anns_field="embedding_value",
                                param=search_params,
                                limit=limit
                            )
                            #print(f"Attribute: {attribute.name}; Type of embedding: {i}; Amount distance values: {len(results[0].distances)}")
                            amount_distances += len(results[0].distances)   
                            test = results[0].distances
                            distances[attribute.name][0]= dict(zip(results[0].ids, results[0].distances))
                            time_sum += time.time() - start_time
                    avg_time = time_sum / rounds
                    avg_dist = amount_distances / rounds
                    output[nprobe][limit] = {
                            "time":avg_time,
                            "amount_distances":avg_dist
                        }
                        
                print(f"NProbe {nprobe}/1024")
                print(time.time() - start_time_)
                
            embedding_collection.release()

    #
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    # with open('vdb_current_match.txt', 'w+') as f:
    #     f.write(s.getvalue())
    
    return output
    print("VDB:--- %s seconds ---" % (time.time() - start_time))
    distance_mat = compute_distances(document_base.nuggets, document_base.attributes)
    print(f"Processed distances VDB: {amount_distances}")
    print(f"Processed distances without VDB: {distance_mat.size}")


def compute_embedding_distances_withoutVDB(path = "corona.bson"):
    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            logger.error("Document base is inconsistent!")
            return

        start_time = time.time()
        distance_mat = compute_distances(document_base.nuggets, document_base.attributes)
    
    return time.time() - start_time ,distance_mat.size