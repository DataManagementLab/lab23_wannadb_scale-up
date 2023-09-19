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
from wannadb.data.signals import CurrentMatchIndexSignal, CombinedEmbeddingSignal
import cProfile, pstats, io
from pstats import SortKey


def test_extraction(path = "C:/Users/Pascal/Desktop\WannaDB/lab23_wannadb_scale-up/cache/exp-2-corona-preprocessed.bson"): 
    pr = cProfile.Profile()
    #Experiment 1 - Vector database for CurrentMatchingIndex

    with open(path, "rb") as file:
        document_base = DocumentBase.from_bson(file.read())

    if not document_base.validate_consistency():
        print("Document base is inconsistent!")
        return

    with vectordb() as vb:
        #times_per_attribute = []
        #vb.extract_nuggets(document_base)

        collection = Collection('Embeddings')
        collection.load()

        embedding_list = []
        attribute = document_base.attributes[0]
        for signal in vb._embedding_identifier:
            if  signal in attribute.signals:
                embedding_list.append(attribute[signal])
            else:
                embedding_list.append(np.zeros(1024))

        if len(embedding_list) > 0:
            combined_embedding =np.concatenate(embedding_list)

        print(len(document_base.nuggets[0][CombinedEmbeddingSignal]))
        print(len(combined_embedding))


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

        '''
        target_embedding = document_base.nuggets[0][CombinedEmbeddingSignal]
        re_doc = document_base.nuggets[0].document
        remaining_docs = []
        for id, i in enumerate(document_base.documents):
            if i != re_doc:
                remaining_docs.append(i)

        print("Updating")
        vb.updating_distances_documents(target_embedding= target_embedding, documents = remaining_docs, document_base=document_base)
        '''

        collection.release()

            

test_extraction()