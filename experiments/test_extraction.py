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
        vb.extract_nuggets(document_base)

        adjusted_collection = Collection('adjusted_embeddings')
        full_collection = Collection('full_embeddings')
        print(f"Anzahl der Nuggets: {adjusted_collection.num_entities}")
        adjusted_collection.load()
        full_collection.load()

        embedding_list = []
        attribute = document_base.attributes[0]
        sample_nugget= document_base.nuggets[0]
        for signal in vb._embedding_identifier:
            if  (signal in attribute.signals) and (signal in sample_nugget.signals):
                embedding_list.append(attribute[signal])
                print(f"Attribute Signal: {signal}")

        if len(embedding_list) > 0:
            combined_embedding =np.concatenate(embedding_list)


        result_docs = vb.compute_inital_distances(combined_embedding,document_base)
        
        for i in result_docs:
          print(f"Nugget: {i[CurrentMatchIndexSignal]} Document: {i.name} Distance: {i.nuggets[i[CurrentMatchIndexSignal]][CachedDistanceSignal]}")


        target_embedding = result_docs[0].nuggets[result_docs[0][CurrentMatchIndexSignal]][CombinedEmbeddingSignal]


        vb.updating_distances_documents(target_embedding, result_docs, document_base)


        with open(path, "rb") as file:
            document_base = DocumentBase.from_bson(file.read())

        if not document_base.validate_consistency():
            print("Document base is inconsistent!")
            return
    

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

        adjusted_collection.release()

            

test_extraction()

'''
        res = collection.query(
            expr = "dbid  == 0",
            offset = 0,
            limit = 10, 
            output_fields = ["id", "document_id", "embedding_value"],
            )
        
        print(len(res))
        print(res[0])
        vbd_value=res[0]['embedding_value']
        sec_value = document_base.nuggets[0][AdjustedCombinedSignal]

        def vectors_are_equal(vector1, vector2):
            # Überprüfe, ob die Längen der Vektoren gleich sind
            if len(vector1) != len(vector2):
                return False
            
            # Vergleiche die Elemente der Vektoren
            for i in range(len(vector1)):
                if vector1[i] != vector2[i]:
                    return False
            
            # Wenn alle Elemente übereinstimmen, gib True zurück
            return True

        print(vectors_are_equal(vbd_value, sec_value))  # True, weil beide Vektoren übereinstimmen

        from sklearn.metrics.pairwise import cosine_distances
        import numpy as np




        embedding_list = []
        attribute = document_base.attributes[0]
        for signal in vb._embedding_identifier:
            if  signal in attribute.signals:
                embedding_list.append(attribute[signal])
                print(f"Attribute Signal1: {signal}")

        if len(embedding_list) > 0:
            combined_embedding =np.concatenate(embedding_list)

              # Um die Cosine-Distanz zu berechnen, müssen die Vektoren als Zeilen in einer Matrix vorliegen.
        # Wir können reshape verwenden, um sie in die richtige Form zu bringen.
        matrix = np.vstack((combined_embedding, vbd_value))

        # Cosine-Distanz berechnen
        cosine_distance = cosine_distances(matrix)

        # Der Wert cosine_distance[0, 1] enthält die Cosine-Distanz zwischen vector_a und vector_b
        print("Cosine-Distanz zwischen VDB value:", cosine_distance[0, 1])

        
        print(f"Dimension nugget {len(document_base.nuggets[0][AdjustedCombinedSignal])}")
        print(f"Dimension Combined Attribute: {len(combined_embedding)}")

        print(f"Combined Attribute: {combined_embedding}")
        print(f"LabelEmbedding Attribute: {attribute[LabelEmbeddingSignal]}")

        print(f"Combined Nugget: {document_base.nuggets[0][AdjustedCombinedSignal]}")
        print(f"LabelEmbedding Nugget: {document_base.nuggets[0][LabelEmbeddingSignal]}")
        '''