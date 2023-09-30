import pandas as pd
from wannadb.data.signals import NaturalLanguageLabelSignal, LabelSignal, CachedContextSentenceSignal, LabelEmbeddingSignal, CombinedEmbeddingSignal, CachedDistanceSignal
import copy
import os
import numpy as np

from pymilvus import (
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from wannadb.configuration import Pipeline
from wannadb.data.data import Attribute, Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import BERTContextSentenceEmbedder,  \
     SBERTTextEmbedder, SBERTLabelEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher, CombineEmbedder
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback
import datasets.corona.corona as corona
import datasets.skyscraper.skyscraper as skyscraper
import datasets.wikipedia.wikipedia as wikipedia
from wannadb.data.vector_database import EMBEDDING_COL_NAME, vectordb
from pymilvus.exceptions import SchemaNotReadyException
from wannadb.matching.distance import SignalsMeanDistance
import time 
from typing import List

def consider_overlap_as_match(true_start, true_end, pred_start, pred_end):
    """Determines whether the predicted span is considered a match of the true span."""
    # considered as overlap if at least half of the larger span
    pred_length = pred_end - pred_start
    true_length = true_end - true_start

    valid_overlap = max(pred_length // 2, true_length // 2, 1)

    if pred_start <= true_start:
        actual_overlap = min(pred_end - true_start, true_length)
    else:
        actual_overlap = min(true_end - pred_start, pred_length)

    return actual_overlap >= valid_overlap


def create_dataframes_attributes_nuggets(document_base: DocumentBase):
    for document in document_base.documents:
        attributes_and_matches_df = pd.DataFrame({
            "attribute": document_base.attributes,  # object ==> cannot be written to csv
            "raw_attribute_name": [attribute.name for attribute in document_base.attributes],
            "nl_attribute_name": [attribute[NaturalLanguageLabelSignal] for attribute in document_base.attributes],
            "matching_nuggets": [document.attribute_mappings[attribute.name] for attribute in
                                    document_base.attributes],  # objects ==> cannot be written to csv
            "matching_nugget_texts": [[n.text for n in document.attribute_mappings[attribute.name]] for attribute in
                                        document_base.attributes]
        })

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        #print(attributes_and_matches_df)

        nuggets_df = pd.DataFrame({
            "nugget": document.nuggets,  # object ==> cannot be written to csv
            "raw_nugget_label": [nugget[LabelSignal] for nugget in document.nuggets],
            "nl_nugget_label": [nugget[NaturalLanguageLabelSignal] for nugget in document.nuggets],
            "nugget_text": [nugget.text for nugget in document.nuggets],
            "context_sentence": [nugget[CachedContextSentenceSignal]["text"] for nugget in document.nuggets],
            "start_char_in_context": [nugget[CachedContextSentenceSignal]["start_char"] for nugget in
                                        document.nuggets],
            "end_char_in_context": [nugget[CachedContextSentenceSignal]["end_char"] for nugget in document.nuggets]
        })

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        #print(nuggets_df)
    return attributes_and_matches_df, nuggets_df

def get_documentbase(dataset: str) -> DocumentBase:
    '''
    Load document collection
    '''
    with ResourceManager() as resource_manager:
        statistics = Statistics(do_collect=True)
        ################################################################################################################
        # dataset
        ################################################################################################################
        if dataset == 'covid-19':
            dataset = corona
        elif dataset == 'skyscrapers':
            dataset == skyscraper
        elif dataset == 'wikipedia':
            dataset == wikipedia
        else:
            raise KeyError(f"Unkown dataset: {dataset}")
        
        documents = dataset.load_dataset()

        statistics["dataset"]["dataset_name"] = dataset.NAME
        statistics["dataset"]["attributes"] = dataset.ATTRIBUTES
        statistics["dataset"]["num_documents"] = len(documents)

        for attribute in dataset.ATTRIBUTES:
            statistics["dataset"]["num_mentioned"][attribute] = 0
            for document in documents:
                if document["mentions"][attribute]:
                    statistics["dataset"]["num_mentioned"][attribute] += 1

        ################################################################################################################
        # document base
        ################################################################################################################
        # select the "user-provided" attribute names and create mappings between them and the dataset's attribute names
        user_attribute_names = dataset.ATTRIBUTES
        statistics["user_provided_attribute_names"] = user_attribute_names
        user_attribute_name2attribute_name = {
            u_attr_name: attr_name for u_attr_name, attr_name in zip(user_attribute_names, dataset.ATTRIBUTES)
        }

        cached_document_base = DocumentBase(
            documents=[Document(doc["id"], doc["text"]) for doc in documents],
            attributes=[Attribute(attribute_name) for attribute_name in user_attribute_names]
        )

        ################################################################################################################
        # preprocessing
        ################################################################################################################
        path = os.path.join(os.path.dirname(__file__), "..", "cache", f"exp-2-{dataset.NAME}-preprocessed.bson")
        if not os.path.isfile(path):

            wannadb_pipeline = Pipeline([
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

            statistics["preprocessing"]["config"] = wannadb_pipeline.to_config()

            wannadb_pipeline(
                document_base=cached_document_base,
                interaction_callback=EmptyInteractionCallback(),
                status_callback=EmptyStatusCallback(),
                statistics=statistics["preprocessing"]
            )


            # Relative path to the desired directory
            relative_path = "cache"

            # Absolute path based on the current directory
            absolute_path = os.path.join(os.getcwd(), relative_path)

            # Check if the directory exists
            if not os.path.exists(absolute_path):
                os.makedirs(absolute_path)
                print(f"'{absolute_path}' was successfully created!")
            else:
                print(f"'{absolute_path}' already exists.")

            path = os.path.join(os.path.dirname(__file__), "..", "cache",
                                f"exp-2-{dataset.NAME}-preprocessed.bson")
            try:
                with open(path, "wb") as file:
                    file.write(cached_document_base.to_bson())
            except Exception as e:
                print("Could not write document base to file.")
                print(e)
        else:
            path = os.path.join(os.path.dirname(__file__), "..", "cache",
                                f"exp-2-{dataset.NAME}-preprocessed.bson")
            with open(path, "rb") as file:
                cached_document_base = DocumentBase.from_bson(file.read())

        for attribute in dataset.ATTRIBUTES:
            statistics["preprocessing"]["results"]["num_extracted"][attribute] = 0
            for document, wannadb_document in zip(documents, cached_document_base.documents):
                match = False
                for mention in document["mentions"][attribute]:
                    for nugget in wannadb_document.nuggets:
                        if consider_overlap_as_match(mention["start_char"], mention["end_char"],
                                                    nugget.start_char, nugget.end_char):
                            match = True
                            break
                if match:
                    statistics["preprocessing"]["results"]["num_extracted"][attribute] += 1   
    document_base = copy.deepcopy(cached_document_base)
    return document_base

def load_test_vdb(document_base: DocumentBase, full_embeddings=False):

    with vectordb() as vb:
        collections = utility.list_collections()

        for collection in collections:
            Collection(collection).drop()

        vb._embedding_value = FieldSchema( 
            name="embedding_value",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,
            )
        
        if full_embeddings:
            vb._embedding_value = FieldSchema( 
            name="embedding_value",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024*3,
            )
                            
        embbeding_schema = CollectionSchema(
                            fields=[vb._dbid,vb._id, vb._document_id, vb._embedding_value],
                            description="Schema for nuggets",
                            enable_dynamic_field=True,
                            )
        
        collection = Collection(
                    name=EMBEDDING_COL_NAME,
                    schema=embbeding_schema,
                    using="default",
                    shards_num=1,
                )
        collection = Collection(EMBEDDING_COL_NAME)

        dbid_counter = 0
        for doc_id, document in enumerate(document_base.documents):
                for id,nugget in enumerate(document.nuggets):

                    embedding = nugget[LabelEmbeddingSignal]

                    if full_embeddings:
                        embedding = nugget[CombinedEmbeddingSignal]

                    data = [
                        [dbid_counter],
                        [id],
                        [doc_id],
                        [embedding],     
                        ]
                    collection.insert(data)

                    dbid_counter = dbid_counter+1

                collection.flush()
                
        #Vector index
        nlist = 4 * int(np.sqrt(dbid_counter))
        vb._index_params["params"]["nlist"] = nlist

        try:
            collection = Collection(EMBEDDING_COL_NAME)
            collection.drop_index()
        except SchemaNotReadyException:
            pass

        collection.create_index(
            field_name='embedding_value', 
            index_params=vb._index_params
            ) 
        
def compute_distances_and_store(document_base, attribute, full_embeddings = False):
    if full_embeddings:
        distance = SignalsMeanDistance(signal_identifiers=["LabelEmbeddingSignal",
                            "TextEmbeddingSignal",
                            "ContextSentenceEmbeddingSignal"])
    else:
        distance = SignalsMeanDistance(signal_identifiers=[LabelEmbeddingSignal])

    overall_start_time = time.perf_counter()
    start_time = time.perf_counter()
    distances = distance.compute_distances(document_base.nuggets, [attribute], Statistics(do_collect=True))
    search_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    for nugget, dist in zip(document_base.nuggets, distances):
        nugget[CachedDistanceSignal] = CachedDistanceSignal(dist)
    storing_time = time.perf_counter()-start_time
    overall_time = time.perf_counter()-overall_start_time
    
    return overall_time, search_time, storing_time
