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
from wannadb.data.signals import LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, RelativePositionSignal, POSTagsSignal
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
    dim=1024,
)
embbeding_schema = CollectionSchema(
    fields=[id, embedding_type, embedding_value],
    description="Schema for nuggets",
    enable_dynamic_field=True,
)

# vector index params

index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}

# distance params
distance_params = {
    "metric": "IP", 
    "dim": 1024
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
                "TextEmbeddingSignal",
                "ContextSentenceEmbeddingSignal",
            ]

        with self:
            for i in utility.list_collections():
                utility.drop_collection(i)

        logger.info("Vector database initialized")
        
        
    

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
                    schema=embbeding_schema,
                    using="default",
                    shards_num=2,
                )
             logger.info("Created collection Embeddings")

        logger.info("Start extracting nuggets from document base")
        collection = Collection("Embeddings")
        for document in documentBase.documents:
                for id,nugget in enumerate(document.nuggets):
                    amount_embeddings = set(self._embedding_identifier).intersection(set(nugget.signals.keys()))
                    if amount_embeddings:
                        data = [
                            [f"{document.name};{str(nugget._start_char)};{str(nugget._end_char)}"]*len(amount_embeddings),
                            [key for key in nugget.signals.keys() if key in amount_embeddings],
                            [nugget.signals[key].value for key in nugget.signals.keys() if key in amount_embeddings],     
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
    
'''  def compute_distance(self, documentBase : DocumentBase) -> dict:
        """
        Compute distance between nuggets
        """
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 0}
        embeddding_signals = ['LabelEmbeddingSignal', 'TextEmbeddingSignal', 'ContextSentenceEmbeddingSignal']

        document_names = [re.sub('[^a-zA-Z0-9 \n\.]', '_', document.name) for document in document_base.documents]    
        
        final_results = {}

        #for every document in the document base
        for document_name in document_names:
            collection = Collection(document_name)
            collection.load()
            document_results= {}

            print(collection.name)
            print(collection.num_entities)
        
            #for every attribute in the document base
            for attribute in attributes:
                distance_results = []

                #for every embedding signal in the attribute
                for i in [signal.identifier for signal in attribute.signals.values() if signal.identifier in embeddding_signals]:
                    #Get the embeddings for the attribute
                    attribute_embeddings= [attribute.signals[i].value]

                    #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
                    results = collection.search(
                         data = attribute_embeddings,
                         anns_field="embedding_value",
                         param=search_params,
                         limit=10,
                         expr = f"embedding_type == '{i}'"
                    )
                    #Create dict with the nugget ids and their distance to the attribute for the current embedding signal
                    distance_results.append(dict(zip(results[0].ids, results[0].distances)))
                    #print(f"Attribute: {attribute.name} Embedding Signal: {i}")
                    #print(distance_results[-1])
                temp = {}
                #print(distance_results)

                #for every distance result of document and attributes
                for i in range(len(distance_results)):
                    temp = mergeDictionary(temp, distance_results[i])
                    #print(f"temp: {temp}")
                document_results[attribute.name] = temp
                final_results[document_name] = document_results
                #print(f"final_results: {final_results}")
        return final_results
                
    def mergeDictionary(dict_1, dict_2):
     dict_3 = {**dict_1, **dict_2}
     for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
            dict_3[key] = [value , dict_1[key]]
     return dict_3
'''




#Tests
def documents() -> List[Document]:
    return [
        Document(
            "document-0",
            "Wilhelm Conrad Röntgen (/ˈrɛntɡən, -dʒən, ˈrʌnt-/; [ˈvɪlhɛlm ˈʁœntɡən]; 27 March 1845 – 10 "
            "February 1923) was a German physicist, who, on 8 November 1895, produced and detected "
            "electromagnetic radiation in a wavelength range known as X-rays or Röntgen rays, an achievement "
            "that earned him the first Nobel Prize in Physics in 1901. In honour of his accomplishments, in "
            "2004 the International Union of Pure and Applied Chemistry (IUPAC) named element 111, "
            "roentgenium, a radioactive element with multiple unstable isotopes, after him."
        ),
        Document(
            "document-1",
            "Wilhelm Carl Werner Otto Fritz Franz Wien ([ˈviːn]; 13 January 1864 – 30 August 1928) was a "
            "German physicist who, in 1893, used theories about heat and electromagnetism to deduce Wien's "
            "displacement law, which calculates the emission of a blackbody at any temperature from the "
            "emission at any one reference temperature. He also formulated an expression for the black-body "
            "radiation which is correct in the photon-gas limit. His arguments were based on the notion of "
            "adiabatic invariance, and were instrumental for the formulation of quantum mechanics. Wien "
            "received the 1911 Nobel Prize for his work on heat radiation. He was a cousin of Max Wien, "
            "inventor of the Wien bridge."
        )
    ]


def create_random_float_vector_dimension_1024() -> List[float]:
    return [random.random() for _ in range(1024)]


def attributes() -> List[Attribute]:
    name_attr = Attribute('name')
    name_attr.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    name_attr.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    month_attr = Attribute('month')
    month_attr.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    month_attr.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    year_attr = Attribute('year')
    year_attr.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    year_attr.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    return [
        name_attr,
        month_attr,
        year_attr
    ]


def information_nuggets(documents) -> List[InformationNugget]:
    nugget_one = InformationNugget(documents[0], 0, 22)
    nugget_one.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_one.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_one.__setitem__(key='LabelSignal', value='test1')
    nugget_two = InformationNugget(documents[0], 56, 123)
    nugget_two.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_two.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_two.__setitem__(key='LabelSignal', value='test2')
    nugget_three = InformationNugget(documents[1], 165, 176)
    nugget_three.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_three.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_three.__setitem__(key='LabelSignal', value='test3')
    nugget_four = InformationNugget(documents[1], 234, 246)
    nugget_four.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_four.__setitem__(key='TextEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_four.__setitem__(key='LabelSignal', value='test4')
    return [
        nugget_one,
        nugget_two,
        nugget_three,
        nugget_four,    
    ]


def document_base(documents, information_nuggets, attributes) -> DocumentBase:
    # link nuggets to documents
    for nugget in information_nuggets:
        nugget.document.nuggets.append(nugget)

    return DocumentBase(
        documents=documents,
        attributes=attributes
    )

documents = documents()
attributes = attributes()
information_nuggets = information_nuggets(documents)
document_base = document_base(documents, information_nuggets, attributes)


search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 0}
embeddding_signals = ['LabelEmbeddingSignal', 'TextEmbeddingSignal', 'ContextSentenceEmbeddingSignal']
        
final_results = {}

with vectordb() as vb:
    for i in utility.list_collections():
                utility.drop_collection(i)

    vb.extract_nuggets(document_base)

    start_time = time.time()
    embedding_collection = Collection("Embeddings")
    embedding_collection.load()
    #dist_collection = Collection("Distances")
    #dist_collection.load()

    distances={}
    #for every attribute in the document base
    for attribute in attributes:
        distances[attribute.name]={}
        #for every embedding signal in the attribute
        for i in [signal.identifier for signal in attribute.signals.values() if signal.identifier in embeddding_signals]:

            #Get the embeddings for the attribute
            attribute_embeddings= [attribute.signals[i].value]

            #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
            results = embedding_collection.search(
                data = attribute_embeddings,
                anns_field="embedding_value",
                param=search_params,
                limit=10,
                expr = f"embedding_type == '{i}'"
                )       
            distances[attribute.name][i]= dict(zip(results[0].ids, results[0].distances))
    print(distances)
print("VDB:--- %s seconds ---" % (time.time() - start_time))


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
        distances: np.ndarray = np.zeros((len(xs), len(ys)))
        for idx in range(3):
            if xs_is_present[idx] == 1 and ys_is_present[idx] == 1:
                x_embeddings: np.ndarray = np.array([x[signal_identifiers[idx]] for x in xs])
                y_embeddings: np.ndarray = np.array([y[signal_identifiers[idx]] for y in ys])
                tmp: np.ndarray = cosine_distances(x_embeddings, y_embeddings)
                distances = np.add(distances, tmp)

        if xs_is_present[3] == 1 and ys_is_present[3] == 1:
            x_positions: np.ndarray = np.array([x[signal_identifiers[3]] for x in xs])
            y_positions: np.ndarray = np.array([y[signal_identifiers[3]] for y in ys])
            tmp: np.ndarray = np.zeros((len(x_positions), len(y_positions)))
            for x_ix, x_value in enumerate(x_positions):
                for y_ix, y_value in enumerate(y_positions):
                    tmp[x_ix, y_ix] = np.abs(x_value - y_value)
            distances = np.add(distances, tmp)

        if xs_is_present[4] == 1 and ys_is_present[4] == 1:
            x_values: list[list[str]] = [x[signal_identifiers[4]] for x in xs]
            y_values: list[list[str]] = [y[signal_identifiers[4]] for y in ys]
            tmp: np.ndarray = np.ones((len(x_values), len(y_values)))
            for x_ix, x_value in enumerate(x_values):
                for y_ix, y_value in enumerate(y_values):
                    if x_value == y_value:
                        tmp[x_ix, y_ix] = 0
            distances = np.add(distances, tmp)

        actually_present: np.ndarray = xs_is_present * ys_is_present
        if np.sum(actually_present) == 0:
            print("Without VDB--- %s seconds ---" % (time.time() - start_time))
            return np.ones_like(distances)
        else:
            print("Without VDB:--- %s seconds ---" % (time.time() - start_time))
            return np.divide(distances, np.sum(actually_present))

compute_distances(document_base.nuggets, document_base.attributes)

