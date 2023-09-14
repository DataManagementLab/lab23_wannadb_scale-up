from wannadb.data.data import Attribute, Document, DocumentBase, InformationNugget
from wannadb.data.vector_database import vectordb
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
    pr = cProfile.Profile()
    #Experiment 1 - Vector database for CurrentMatchingIndex

    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            print("Document base is inconsistent!")
            return
        
        with vectordb() as vb:
            #times_per_attribute = []

            vb.extract_nuggets(document_base)

            start_time = time.time()
            embedding_collection = Collection("Embeddings")
            embedding_collection.load()
            print(f"Embedding collection has {embedding_collection.num_entities} embeddings.")
            print("Load VDB:--- %s seconds ---" % (time.time() - start_time))
            remaining_documents: List[Tuple(Document, float)] = []

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


            remaining_documents: List[Document] = []
            embedding_collection = Collection('Embeddings')

            search_params = {
            "metric_type": "L2", 
            "offset": 0, 
            "ignore_growing": False, 
            "params": {"nprobe": 100}
            }

            pr.enable()

                    
            results = embedding_collection.search(
                data=[combined_embedding], 
                anns_field="embedding_value", 
                param=search_params,
                limit=100,
                expr= None,
                output_fields=['id'],
                consistency_level="Strong"
            )
            '''
                if results[0].ids: 
                    i[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(results[0][0].id) 
                    i.nuggets[results[0][0].id][CachedDistanceSignal] = CachedDistanceSignal(results[0][0].distance)
                    remaining_documents.append(i)
            '''
            embedding_collection.load()
                    
    print("VDB intial distances:--- %s seconds ---" % (time.time() - start_time))
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    with open('vdb_current_match.txt', 'w+') as f:
        f.write(s.getvalue())

    
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

