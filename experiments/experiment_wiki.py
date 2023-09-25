import copy
import json
import logging.config
import os
import random
import time
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from wannadb.configuration import Pipeline
from wannadb.data.data import Attribute, Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.matching.distance import SignalsMeanDistance
from wannadb.matching.matching import RankingBasedMatcher, RankingBasedMatcherVDB
from wannadb.preprocessing.embedding import BERTContextSentenceEmbedder, FastTextLabelEmbedder, \
    RelativePositionEmbedder, SBERTTextEmbedder, SBERTExamplesEmbedder, SBERTLabelEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher, CombineEmbedder
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback
import datasets.wikipedia.wikipedia as dataset
from experiments.automatic_feedback import AutomaticRandomRankingBasedMatchingFeedback
from experiments.baselines.baseline_bart_seq2seq import calculate_f1_scores
from experiments.util import consider_overlap_as_match
from wannadb.data.vector_database import EMBEDDING_COL_NAME, vectordb
from wannadb.data.signals import UserProvidedExamplesSignal, LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal
import cProfile, pstats, io
from pstats import SortKey

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

RESULTS_FILENAME = r"exp-2.json"

def run_experiment_wiki(index_types: List[str] = ["FLAT","IVF_FLAT","IVF_SQ8","GPU_IVF_FLAT"]):

    with ResourceManager() as resource_manager:
        statistics = Statistics(do_collect=True)
        index_type_durations = []
        ################################################################################################################
        # dataset
        ################################################################################################################
        documents = dataset.load_dataset("medium_large")

        statistics["dataset"]["dataset_name"] = dataset.NAME
        statistics["dataset"]["attributes"] = dataset.ATTRIBUTES
        statistics["dataset"]["num_documents"] = len(documents)

        for attribute in dataset.ATTRIBUTES:
            statistics["dataset"]["num_mentioned"][attribute] = 0

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
                    

        ################################################################################################################
        # Load embeddings into vector database
        ################################################################################################################
        print("Start writing in VDB")
        start_time = time.time()
        with vectordb() as vdb:
            vdb.extract_nuggets(cached_document_base)
        extracting_nugget_time = time.time() - start_time
        print("Finished writing in VDB:--- %s seconds ---" % (extracting_nugget_time))
        

        for index_type in index_types:
            
            ################################################################################################################
            # matching phase
            ################################################################################################################

            for attribute_name in dataset.ATTRIBUTES:
                statistics["matching"]["results"]["considered_as_match"][attribute_name] = set()

            # random seeds have been randomly chosen once from [0, 1000000]
            random_seeds = [200488]# 422329, 449756]#, 739608, 983889], 836016, 264198, 908457, 205619, 461905]
            #random_seeds = [794009, 287762, 880883, 663238, 137616, 543468, 329177, 322737, 343909, 824474, 220481,
            #               832096,
            #                962731, 345784, 317557, 696622, 675696, 467273, 475463, 540128]
            

            pr = cProfile.Profile()
            pr.enable()
            start_time = time.time()
             
            with vectordb() as vdb:
                collection = Collection(EMBEDDING_COL_NAME)
                try:
                    collection.release()
                except:
                    pass
                
                vdb.regenerate_index(index_type)
                collection.load()

                for run, random_seed in enumerate(random_seeds):
                    # load the document base
                    document_base = copy.deepcopy(cached_document_base)
                    
                    print("\n\n\nExecuting run {}.".format(run + 1))

                    # load the document base
                    # path = os.path.join(os.path.dirname(__file__), "..", "cache", f"exp-2-{dataset.NAME}-preprocessed.bson")
                    # with open(path, "rb") as file:
                    #     document_base = DocumentBase.from_bson(file.read())

                        
                    wannadb_pipeline = Pipeline(
                        [
                            ContextSentenceCacher(),
                            RankingBasedMatcherVDB(
                                max_num_feedback=10,
                                len_ranked_list=10,
                                max_distance=0.2,
                                num_random_docs=1,
                                sampling_mode="AT_MAX_DISTANCE_THRESHOLD",
                                vector_database = vdb,
                                adjust_threshold=True,
                                embedding_identifier=[
                                                "LabelEmbeddingSignal",
                                                "TextEmbeddingSignal",
                                                "ContextSentenceEmbeddingSignal"

                                ],

                                nugget_pipeline=Pipeline(
                                    [
                                        ContextSentenceCacher(),
                                        CopyNormalizer(),
                                        OntoNotesLabelParaphraser(),
                                        SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                                        SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                                        CombineEmbedder()
                                    ]
                                )
                            )
                        ]
                        )

                    '''
                    wannadb_pipeline = Pipeline([
                    ContextSentenceCacher(),
                    RankingBasedMatcher(
                        distance=SignalsMeanDistance(
                            signal_identifiers=[
                                "LabelEmbeddingSignal",
                                "TextEmbeddingSignal",
                                "ContextSentenceEmbeddingSignal"
                            ]
                        ),
                        max_num_feedback=10,
                        len_ranked_list=10,
                        max_distance=0.2,  
                        num_random_docs=1,
                        sampling_mode="AT_MAX_DISTANCE_THRESHOLD",  
                        adjust_threshold=True,
                        nugget_pipeline=Pipeline(
                                [
                                    ContextSentenceCacher(),
                                    CopyNormalizer(),
                                    OntoNotesLabelParaphraser(),
                                    SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                                    SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
                                ]
                            )
                        )
                        ])
                        '''

                    statistics["matching"]["config"] = wannadb_pipeline.to_config()

                    # set the random seed
                    random.seed(random_seed)

                    logger.setLevel(logging.WARN)

                    wannadb_pipeline(
                    document_base=document_base,
                    interaction_callback=AutomaticRandomRankingBasedMatchingFeedback(
                        documents,
                        user_attribute_name2attribute_name
                    ),
                    status_callback=EmptyStatusCallback(),
                    statistics=statistics["matching"]["runs"][str(run)]
                    )
                    
                collection.release()

            duration = time.time() - start_time
            index_type_durations.append(duration)
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            with open(path[:-5] +f'automatch_vdb_single_{index_type}.txt', 'w+') as f:
                f.write(s.getvalue()) 

        ################################################################################################################
        # Duration per index type
        ################################################################################################################
        try:
            _, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(x=index_types, y=index_type_durations, ax=ax, color="#0c2461")
            ax.set_ylabel("Duration in seconds")
            ax.set_title("Durations per Index Type", size=12)
            ax.tick_params(axis="x", labelsize=7)
            plt.xticks(rotation=20, ha='right')
            plt.subplots_adjust(0.09, 0.15, 0.99, 0.94)

            for inex_type, duration in zip(np.arange(len(index_types)), index_type_durations):
                ax.text(
                    inex_type,
                    duration,
                    str(round(duration, 2)),
                    fontsize=9,
                    horizontalalignment="center"
                )

            plt.savefig(path[:-5] + f"-durations-{index_type}.pdf", format="pdf", transparent=True)
        except:
            print("No durations plot")
        print("Extracting nuggets to VDB:--- %s seconds ---" % (extracting_nugget_time))