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
from typing import List, Any
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
import datasets.corona.corona as dataset
from wannadb.data.signals import CachedDistanceSignal
from wannadb.data.signals import CurrentMatchIndexSignal, NewCurrentMatchIndexSignal
import cProfile, pstats, io
from pstats import SortKey


with ResourceManager() as resource_manager:
    documents = dataset.load_dataset()
    document_base = DocumentBase(documents=[Document(doc['id'], doc['text']) for doc in documents], 
                             attributes=[Attribute(attribute) for attribute in dataset.ATTRIBUTES])

    # preprocess the data
    default_pipeline = Pipeline([
                        StanzaNERExtractor(),
                        SpacyNERExtractor("SpacyEnCoreWebLg"),
                        ContextSentenceCacher(),
                        CopyNormalizer(),
                        OntoNotesLabelParaphraser(),
                        SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                        SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
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
    print (f"Document base has {len(document_base.documents)} documents.")
    print (f"Document base has {len(document_base.nuggets)} nuggets.")
    print (f"Document base has {len(document_base.attributes)} attributes.")
    print(f"Document base has {len([nugget for nugget in document_base.nuggets if 'LabelEmbeddingSignal' in nugget.signals.keys()])} nugget LabelEmbeddings.")
    print(f"Document base has { len([attribute for attribute in document_base.attributes if 'LabelEmbeddingSignal' in attribute.signals.keys()])} attribute LabelEmbeddings.")

pr = cProfile.Profile()
#Experiment 1 - Vector database for CurrentMatchingIndex
with vectordb() as vb:
        for i in utility.list_collections():
                utility.drop_collection(i)

        vb.extract_nuggets(document_base)
        #times_per_attribute = []

        pr.enable()
        #start_time = time.time()
        embedding_collection = Collection("Embeddings")
        embedding_collection.load()
        print(f"Embedding collection has {embedding_collection.num_entities} embeddings.")
        
        amount_distances = 0
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
                    'param' : {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 0},
                    'limit' : 1,
                    'expr' : f"embedding_type == 'LabelEmbeddingSignal' and id like \"{i.name}%\""
                }
                results = embedding_collection.search(**search_param)
                if results[0].ids: 
                    i[NewCurrentMatchIndexSignal] = NewCurrentMatchIndexSignal(results[0][0].id)
                #times_per_attribute.append(time.time()-time_per_attribute) 
                amount_distances = amount_distances+1
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
with open('vdb_current_match.txt', 'w+') as f:
    f.write(s.getvalue())

              

#print("VDB overall:--- %s seconds ---" % (time.time() - start_time))
#print("VDB attribute average: --- %s seconds ---" % (np.mean(times_per_attribute)))
#print(f"Processed distances VDB: {amount_distances}")

#Experiment 2 - Without Vector database
#times_per_attribute = []
#start_time = time.time()
pr = cProfile.Profile()
pr.enable()
for attribute in document_base.attributes:
    #time_per_attribute = time.time()

    statistics = Statistics(do_collect=True)
    distances: np.ndarray = SignalsMeanDistance(signal_identifiers=["LabelEmbeddingSignal"]).compute_distances([attribute], document_base.nuggets, statistics["distance"])[0]
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
#print("Without VDB overall:--- %s seconds ---" % (time.time() - start_time))
#print("Without VDB attribute average: --- %s seconds ---" % (np.mean(times_per_attribute)))
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
with open('old_current_match.txt', 'w+') as f:
    f.write(s.getvalue())