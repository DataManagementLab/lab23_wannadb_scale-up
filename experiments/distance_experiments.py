from wannadb.data.data import Attribute, Document, DocumentBase, InformationNugget
from wannadb.data.vector_database import vectordb, VECTORDB
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List, Any, Tuple
import time
import numpy as np
from wannadb.statistics import Statistics
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances


from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTTextEmbedder, SBERTLabelEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback
from wannadb.matching.distance import SignalsMeanDistance
#import datasets.corona.corona as dataset
from wannadb.data.signals import CachedDistanceSignal
from wannadb.data.signals import CurrentMatchIndexSignal, NewCurrentMatchIndexSignal
import cProfile, pstats, io
from pstats import SortKey


def new_compute_embedding_distances(path = "corona.bson"): 
    pr = cProfile.Profile()
    #Experiment 1 - Vector database for CurrentMatchingIndex

    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            print("Document base is inconsistent!")
            return
        
        with vectordb() as vb:
            #times_per_attribute = []

            pr.enable()
            start_time = time.time()
            embedding_collection = Collection("Embeddings")
            embedding_collection.load()
            print(f"Embedding collection has {embedding_collection.num_entities} embeddings.")
            print("Load VDB:--- %s seconds ---" % (time.time() - start_time))
            remaining_documents: List[Tuple(Document, float)] = []


            #for every attribute in the document base
            for attribute in document_base.attributes:
                #time_per_attribute = time.time()
                
                statistics = Statistics(do_collect=True)
                distances: np.ndarray = SignalsMeanDistance( signal_identifiers=["LabelEmbeddingSignal"]).compute_distances([attribute], document_base.nuggets, statistics["distance"])[0]
                for nugget, distance in zip(document_base.nuggets, distances):
                    nugget[CachedDistanceSignal] = CachedDistanceSignal(distance)

                for i in document_base.documents:

                    #Get the embeddings for the attribute
                    attribute_embeddings= [attribute.signals['LabelEmbeddingSignal'].value]

                    #Compute the distance between the embeddings of the attribute and the embeddings of the nuggets
                    search_param= {
                        'data' : attribute_embeddings,
                        'anns_field' : "embedding_value",
                        'param' : {"metric_type": "L2", "params": {"nprobe": 1000}, "offset": 0},
                        'limit' : 1,
                        'expr' : f"embedding_type == 'LabelEmbeddingSignal' and id like \"{i.name}%\""
                    }
                    results = embedding_collection.search(**search_param)
                    if results[0].ids: 
                        i[NewCurrentMatchIndexSignal] = NewCurrentMatchIndexSignal(results[0][0].id)
                        remaining_documents.append((i, results[0][0].distance))
            print("VDB intial distances:--- %s seconds ---" % (time.time() - start_time))
            num_feedback: int = 0
            continue_matching: bool = True

