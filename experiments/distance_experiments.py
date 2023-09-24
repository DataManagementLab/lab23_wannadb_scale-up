from wannadb.data.data import Attribute, Document, DocumentBase, InformationNugget
from wannadb.data.vector_database import EMBEDDING_COL_NAME, vectordb
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
from wannadb.data.signals import CurrentMatchIndexSignal
import cProfile, pstats, io
from pstats import SortKey


def new_compute_embedding_distances(path = "C:/Users/Pascal/Desktop\WannaDB/lab23_wannadb_scale-up/cache/exp-2-corona-preprocessed.bson"): 
    #Experiment 1 - Vector database for CurrentMatchingIndex

    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            print("Document base is inconsistent!")
            return
        
        with vectordb() as vb:
            #times_per_attribute = []

            #vb.extract_nuggets(document_base)
            embedding_collection = Collection(EMBEDDING_COL_NAME)

            start_time = time.time()
            embedding_collection.load()
            print(f"Embedding collection has {embedding_collection.num_entities} embeddings.")
            print("Load VDB:--- %s seconds ---" % (time.time() - start_time))

            attribute = document_base.attributes[0]
            signals = ['LabelEmbeddingSignal', 'TextEmbeddingSignal', 'ContextSentenceEmbeddingSignal']

            embedding_list = []
            for signal in signals:
                if signal in attribute.signals:
                    embedding_list.append(attribute[signal])
                else:
                    embedding_list.append(np.zeros(1024))


            if len(embedding_list) > 0:
                combined_embedding =np.concatenate(embedding_list)

            result_docs = vb.compute_inital_distances(attribute_embedding= combined_embedding, document_base=document_base)
            for i in result_docs:
                print(f"Nugget: {i[CurrentMatchIndexSignal]} Document: {i.name} Distance: {i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal]}")
            embedding_collection.release()        


    start_time = time.time()
    attribute = document_base.attributes[0]
        #time_per_attribute = time.time()

    statistics = Statistics(do_collect=True)
    distances: np.ndarray = SignalsMeanDistance(signal_identifiers=['LabelEmbeddingSignal', 'TextEmbeddingSignal', 'ContextSentenceEmbeddingSignal']).compute_distances([attribute], document_base.nuggets, statistics["distance"])[0]
    for nugget, distance in zip(document_base.nuggets, distances):
        nugget[CachedDistanceSignal] = CachedDistanceSignal(distance)

    for document in document_base.documents:
        try:
            index, _ = min(enumerate(document.nuggets), key=lambda nugget: nugget[1][CachedDistanceSignal])
        except ValueError:  # document has no nuggets
            document.attribute_mappings[attribute.name] = []
            statistics[attribute.name]["num_document_with_no_nuggets"] += 1
        else:
            document[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(index)

        #times_per_attribute.append(time.time()-time_per_attribute) 
    print("Without VDB overall:--- %s seconds ---" % (time.time() - start_time))

new_compute_embedding_distances()

