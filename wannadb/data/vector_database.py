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
from typing import List, Any, Optional, Union, Dict
import logging
import time
from wannadb.data.data import DocumentBase, Attribute, Document, InformationNugget
import numpy as np
from wannadb.statistics import Statistics
from wannadb.data.signals import LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, RelativePositionSignal,UserProvidedExamplesSignal, \
    CachedDistanceSignal, CurrentMatchIndexSignal, CombinedEmbeddingSignal, POSTagsSignal, AdjustedCombinedSignal
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances


from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTTextEmbedder, SBERTExamplesEmbedder,BERTContextSentenceEmbedder, SBERTLabelEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher, CombineEmbedder
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
        self._embedding_identifier = ['LabelEmbeddingSignal', 'TextEmbeddingSignal', 'ContextSentenceEmbeddingSignal']
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
            dtype=DataType.INT64,
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
        "metric_type":"COSINE",
        "index_type":"FLAT"
        }

        self._search_params = {
        "metric_type": "COSINE", 
        "offset": 0, 
        "ignore_growing": False, 
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
        if "full_embeddings" not in utility.list_collections():
            full_collection = Collection(
                    name="full_embeddings",
                    schema=self._embbeding_schema,
                    using="default",
                    shards_num=1,
                )
            logger.info("Created collection Embeddings")

        if "adjusted_embeddings" not in utility.list_collections():
             
            sample_nugget = documentBase.nuggets[0]
            sample_attribute = documentBase.attributes[0]

            embeddings_nugget = set(self._embedding_identifier).intersection(sample_nugget.signals)
            embeddings_attribute = set(self._embedding_identifier).intersection(sample_attribute.signals)
            final_embeddings = embeddings_nugget.intersection(embeddings_attribute)

            required_dim = len(final_embeddings)
             
            adjusted_embedding_value = FieldSchema( 
            name="embedding_value",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024*required_dim,
            )

            print(f"Required_dim: {required_dim}")

            adjusted_embbeding_schema = CollectionSchema(
            fields=[self._id, self._document_id, adjusted_embedding_value],
            description="Schema for nuggets",
            enable_dynamic_field=True,
            )
   
            adjusted_collection = Collection(
                    name="adjusted_embeddings",
                    schema=adjusted_embbeding_schema,
                    using="default",
                    shards_num=1,
                )
            logger.info("Created collection Embeddings")

        logger.info("Start extracting nuggets from document base")
        full_collection = Collection("full_embeddings")
        adjusted_collection = Collection("adjusted_embeddings")
        for doc_id, document in enumerate(documentBase.documents):
                document.set_index(doc_id)
                for id,nugget in enumerate(document.nuggets):

                    combined_embedding = nugget[CombinedEmbeddingSignal]
                    print(f"Dim full embedding: {len(combined_embedding)}")
                    data = [
                        [id],
                        [doc_id],
                        [combined_embedding],     
                        ]
                    full_collection.insert(data)
                    logger.info(f"Inserted nugget {id} {combined_embedding} from document {document.name} into full_collection")

                    adjusted_combined_embedding = nugget[AdjustedCombinedSignal]
                    print(f"Dimension Nugget: {len(nugget[AdjustedCombinedSignal])}")
                    data = [
                        [id],
                        [doc_id],
                        [adjusted_combined_embedding],     
                        ]
                    adjusted_collection.insert(data)
                    logger.info(f"Inserted nugget {id} {adjusted_combined_embedding} from document {document.name} into adjusted_collection")

                full_collection.flush()
                adjusted_collection.flush()
        logger.info("Embedding insertion finished")

        #Vector index
        logger.info("Start indexing")
        full_collection.create_index(
            field_name='embedding_value', 
            index_params=self._index_params
            )   
        
        adjusted_collection.create_index(
            field_name='embedding_value', 
            index_params=self._index_params
            )  
        logger.info("Indexing finished")
        logger.info("Extraction finished")

    def compute_inital_distances(self, attribute_embedding : List[float], document_base: DocumentBase) -> List[Document]:

        remaining_documents: List[Document] = []
        adjusted_embedding_collection = Collection('adjusted_embeddings')
 
        results = adjusted_embedding_collection.search(
            data=[attribute_embedding], 
            anns_field="embedding_value", 
            param=self._search_params,
            limit=1,
            expr= None,
            output_fields=['id','document_id'],
            consistency_level="Strong"
        )


        for i in results[0]:
            #print(f"Result[i]: f{i}")
            
            current_document = document_base.documents[i.entity.get('document_id')]
            if not current_document in remaining_documents:
                current_nugget = i.id

                current_document[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(current_nugget)
                print(f"Computed Distance: {i.distance}")
                current_document.nuggets[current_nugget][CachedDistanceSignal] = CachedDistanceSignal(i.distance)
                remaining_documents.append(current_document)
                #print(f"Appended nugget: {current_nugget}; To document {current_document.name} cached index: {current_document[CurrentMatchIndexSignal]}; Cached distance {current_document.nuggets[current_nugget][CachedDistanceSignal]}")
                logger.info(f"Appended nugget: {current_nugget}; To document {current_document.name} cached index: {current_document[CurrentMatchIndexSignal]}; Cached distance {current_document.nuggets[current_nugget][CachedDistanceSignal]}")
            else:
                #print(f"Skip Nugget {i.id}")
                continue
        print(f"Insgesamt: {len(remaining_documents)}")

        return remaining_documents
    
    def updating_distances_documents(self, target_embedding: List[float], documents: List[Document], document_base : DocumentBase):
        full_embedding_collection = Collection('full_embeddings')
        doc_indexes = [doc.get_index() for doc in documents]
        processed = [] 

        #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
        results = full_embedding_collection.search(
            data=[target_embedding], 
            anns_field="embedding_value", 
            param=self._search_params,
            limit=16384,
            expr= f"document_id in {str(doc_indexes)}",
            output_fields=['id','document_id'],
            consistency_level="Strong"
        )

        #print(f"Results[0]: {results[0]}")

        for i in results[0]:
            current_document = document_base.documents[i.entity.get('document_id')]
            #print(f"Current Document: {current_document}")

            if not current_document in processed:
                #print(F"Not already processed")
                current_nugget = i.id
                #print(f"Current Nugget: {current_nugget}")
                if 'CurrentMatchIndexSignal' in current_document.signals:
                    #print(f"Document: {current_document.name} contains CurrentMatchIndex: {current_document[CurrentMatchIndexSignal]}")
                    if i.distance < current_document.nuggets[current_document[CurrentMatchIndexSignal]][CachedDistanceSignal]:
                        #print(f"Neue Distanz: {i.distance} kleiner als alte Distanz: {current_document.nuggets[current_document[CurrentMatchIndexSignal]][CachedDistanceSignal]}")
                        current_document[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(current_nugget)
                        current_document.nuggets[current_nugget][CachedDistanceSignal] = CachedDistanceSignal(i.distance)
                        #print(f"Neuer Currentindex: {current_document[CurrentMatchIndexSignal]}, Neue Distanz: {current_document.nuggets[current_nugget][CachedDistanceSignal]}")
                    else:
                        #print(f"Neue Distanz: {i.distance} größer als alte Distanz: {current_document.nuggets[current_document[CurrentMatchIndexSignal]][CachedDistanceSignal]}")
                        continue
                else:
                    #print("CurrentIndex noch nicht drin")
                    current_document[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(current_nugget)
                    current_document.nuggets[current_nugget][CachedDistanceSignal] = CachedDistanceSignal(i.distance)
                    #print(f"Neuer Current Index: { current_document[CurrentMatchIndexSignal]}; Neue Cached Distanz: {current_document.nuggets[current_nugget][CachedDistanceSignal]}")
            else:
                #print(f"Already processed!")
                continue

    def regenerate_index(self, index_params, collection_name = "Embeddings"):
        collection = Collection(collection_name)
        collection.drop_index()
        collection.create_index(
            field_name='embedding_value', 
            index_params=index_params
            )

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
                SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
                SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                BERTContextSentenceEmbedder("BertLargeCasedResource"),
                CombineEmbedder()
            ])


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
        print(f"Document base has {len([nugget for nugget in document_base.nuggets if 'CombinedEmbeddingSignal' in nugget.signals.keys()])} nugget combined Embeddings")
        print(f"Combined Nugget has {len(document_base.nuggets[0][CombinedEmbeddingSignal])} Dimensions")
        print(f"Adjusted Combined Nugget has {len(document_base.nuggets[0][AdjustedCombinedSignal])} Dimensions")

            
        with open("corona.bson", "wb") as file:
            file.write(document_base.to_bson())



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


def generate_new_index(index_params):
    with vectordb() as vb:
        vb.regenerate_index(index_params)

def compute_embedding_distances(path = "corona.bson", rounds= 1, nprobe_max= 500, max_limit=2001 ):    
    
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

    print("VDB:--- %s seconds ---" % (time.time() - start_time))
    distance_mat = compute_distances(document_base.nuggets, document_base.attributes)
    print(f"Processed distances VDB: {amount_distances}")
    print(f"Processed distances without VDB: {distance_mat.size}")
    return output


def compute_embedding_distances_withoutVDB(path = "corona.bson"):
    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            logger.error("Document base is inconsistent!")
            return

        start_time = time.time()
        distance_mat = compute_distances(document_base.nuggets, document_base.attributes)
    
    return time.time() - start_time ,distance_mat.size


#generate_and_store_embedding('C:\\Users\\Pascal\\Desktop\\WannaDB\\lab23_wannadb_scale-up\\datasets\\corona\\raw-documents')
 
#print(compute_embedding_distances_withoutVDB())
