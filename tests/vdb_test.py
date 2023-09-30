from typing import List
import pytest


from wannadb.data.data import Attribute, Document, DocumentBase, InformationNugget
from wannadb.data.data import Document, DocumentBase
import random
from wannadb.data.vector_database import EMBEDDING_COL_NAME, compute_embedding_distances, compute_embedding_distances_withoutVDB, generate_and_store_embedding, vectordb
from pymilvus import Collection, utility
import re
import xlsxwriter
from wannadb.resources import ResourceManager
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
import pandas as pd
from wannadb.data.signals import NaturalLanguageLabelSignal, LabelSignal, CachedContextSentenceSignal, LabelEmbeddingSignal, CombinedEmbeddingSignal, CachedDistanceSignal
import copy
import os
import numpy as np
from experiments.util import compute_distances_and_store, load_test_vdb

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


from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTTextEmbedder, SBERTLabelEmbedder, BERTContextSentenceEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher, CombineEmbedder
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback
from wannadb.matching.distance import SignalsMeanDistance
import datasets.corona.corona as dataset
import os
from sklearn.preprocessing import normalize

#import datasets.corona.corona as dataset
from wannadb.data.signals import CachedDistanceSignal
from wannadb.data.signals import CurrentMatchIndexSignal, CombinedEmbeddingSignal,  LabelEmbeddingSignal
import cProfile, pstats, io
from pstats import SortKey


def test_document_base_extraction() -> DocumentBase:
    with ResourceManager() as resource_manager:
        statistics = Statistics(do_collect=True)
        ################################################################################################################
        # dataset
        ################################################################################################################
        
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
    document_base = copy.deepcopy(cached_document_base)
    return document_base

def test_vdb_nugget_extraction(document_base):
    load_test_vdb(document_base=document_base,full_embeddings=True)

    with vectordb() as vb:

        collection = Collection(EMBEDDING_COL_NAME)
        collection.load()

        print(collection.num_entities)

        res = collection.query(
            expr = "dbid  < 10",
            offset = 0,
            limit = 10, 
            output_fields = ["id", "document_id", "embedding_value"],
            )
          
        def vectors_are_equal(vector1, vector2):
                # Check if vector length is equal
                assert len(vector1) == len(vector2)
                
                # Check if vector elements are equal
                for i in range(len(vector1)):
                    assert vector1[i] == vector2[i]
        
        for i in range(10):
            nugget = document_base.documents[res[i]['document_id']].nuggets[res[i]['id']]
            vbd_value=res[i]['embedding_value']
            sec_value = nugget[CombinedEmbeddingSignal]
            vectors_are_equal(vbd_value, sec_value)
            assert res[i]['document_id'] == nugget.document.index

        assert collection.num_entities == len(document_base.nuggets)

        collection.release()
    print("True")


def test_vector_search(document_base):
    load_test_vdb(document_base=document_base,full_embeddings=False)

    with vectordb() as vb:
         collection = Collection(EMBEDDING_COL_NAME)
         collection.load()

         attribute = document_base.attributes[0]
         attribute_embedding = attribute[LabelEmbeddingSignal]

         results = collection.search(
                data=[attribute_embedding], 
                anns_field="embedding_value", 
                param=vb._search_params,
                limit=10,
                expr= None,
                output_fields=['id','document_id'],
                consistency_level="Strong"
            )
         
         compute_distances_and_store(document_base, attribute, full_embeddings=False)

         for i in results[0]:
            current_document = document_base.documents[i.entity.get('document_id')]
            current_nugget = i.entity.get('id')
            assert 1-i.distance == current_document.nuggets[current_nugget][CachedDistanceSignal]
    print("True")
             



document_base = test_document_base_extraction()
test_vdb_nugget_extraction(document_base)
test_vector_search(document_base)