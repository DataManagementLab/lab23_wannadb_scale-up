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
from wannadb.preprocessing.other_processing import ContextSentenceCacher, CombineEmbedder
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback
from wannadb.matching.distance import SignalsMeanDistance

#import datasets.corona.corona as dataset
from wannadb.data.signals import CachedDistanceSignal
from wannadb.data.signals import CurrentMatchIndexSignal, CombinedEmbeddingSignal, AdjustedCombinedSignal, LabelEmbeddingSignal
import cProfile, pstats, io
from pstats import SortKey


def test_extraction(path = "C:/Users/Pascal/Desktop\WannaDB/lab23_wannadb_scale-up/cache/corona.bson"): 
    pr = cProfile.Profile()
    #Experiment 1 - Vector database for CurrentMatchingIndex

    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

    if not document_base.validate_consistency():
        print("Document base is inconsistent!")
        return
    
    #print(document_base.nuggets[0][AdjustedCombinedSignal])
    #print(len(document_base.nuggets[0][AdjustedCombinedSignal]))
    #print(len(document_base.nuggets[0][CombinedEmbeddingSignal]))

    with vectordb() as vb:
        #times_per_attribute = []
        #vb.extract_nuggets(document_base)

        collection = Collection('adjusted_embeddings')
        print(f"Anzahl der Nuggets: {collection.num_entities}")
        collection.load()

        '''
        res = collection.query(
            expr = "id == 0",
            offset = 0,
            limit = 100, 
            output_fields = ["id", "document_id", "embedding_value"],
            )
        
        print(res)
        '''

        embedding_list = []
        attribute = document_base.attributes[0]
        for signal in vb._embedding_identifier:
            if  signal in attribute.signals:
                embedding_list.append(attribute[signal])
                print(f"Attribute Signal1: {signal}")

        if len(embedding_list) > 0:
            combined_embedding =np.concatenate(embedding_list)

        print(f"Dimension nugget {len(document_base.nuggets[0][AdjustedCombinedSignal])}")
        print(f"Dimension Combined Attribute: {len(combined_embedding)}")

        print(f"Combined Attribute: {combined_embedding}")
        print(f"LabelEmbedding Attribute: {attribute[LabelEmbeddingSignal]}")

        print(f"Combined Nugget: {document_base.nuggets[0][AdjustedCombinedSignal]}")
        print(f"LabelEmbedding Nugget: {document_base.nuggets[0][LabelEmbeddingSignal]}")

        result_docs = vb.compute_inital_distances(combined_embedding,document_base)

        #print([i[CurrentMatchIndexSignal] for i in result_docs])

        for i in result_docs:
          print(f"Nugget: {i[CurrentMatchIndexSignal]} Document: {i.name} Distance: {i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal]}")


        with open(path, "rb") as file:
            document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            print("Document base is inconsistent!")
            return
    
        for id,i in enumerate(document_base.documents):
            i.set_index(id)

        remaining_documents = []
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
                remaining_documents.append(document)

        remaining_documents = list(sorted(
                    remaining_documents,
                    key=lambda x: x.nuggets[x[CurrentMatchIndexSignal]][CachedDistanceSignal],
                    reverse=True
                ))

        for i in remaining_documents:
            print(f"WithoutVDB - Nugget: {i[CurrentMatchIndexSignal]} Document: {i.name} Distance: {i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal]}")

        collection.release()

            

test_extraction()