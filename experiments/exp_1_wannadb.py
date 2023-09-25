import collections
import logging.config
import random

from aset.configuration import ASETPipeline
from aset.data.data import ASETAttribute, ASETDocument, ASETDocumentBase
from aset.interaction import EmptyInteractionCallback, InteractionCallback
from aset.matching.distance import SignalsMeanDistance
from aset.matching.matching import RankingBasedMatcher
from aset.preprocessing.embedding import BERTContextSentenceEmbedder, FastTextLabelEmbedder, \
    RelativePositionEmbedder, SBERTTextEmbedder
from aset.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from aset.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from aset.preprocessing.normalization import CopyNormalizer
from aset.preprocessing.other_processing import ContextSentenceCacher
from aset.resources import ResourceManager
from aset.statistics import Statistics
from aset.status import EmptyStatusCallback
from experiments.automatic_feedback import AutomaticRandomRankingBasedMatchingFeedback
from experiments.util import consider_overlap_as_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":

    with ResourceManager() as resource_manager:
        statistics = Statistics(do_collect=True)

        ################################################################################################################
        # dataset
        ################################################################################################################

        DATASET = "nobel"  # TODO: choose data set by changing 'nobel' to 'nobel' or 'countries'

        if DATASET == "nobel":
            import datasets.nobel_country.nobel_country as dataset

            ATTRIBUTE_NAME = "country"
        elif DATASET == "countries":
            import datasets.countries_continent.countries_continent as dataset

            ATTRIBUTE_NAME = "continent"
        else:
            assert False, "Unknown dataset!"

        print(f"Dataset: {dataset.NAME}")
        print(f"Attribute: {ATTRIBUTE_NAME}")

        ################################################################################################################
        # execute extraction and matching
        ################################################################################################################

        # load the data
        documents = dataset.load_dataset()
        document_base = ASETDocumentBase(
            documents=[ASETDocument(doc["id"], doc["text"]) for doc in documents],
            attributes=[ASETAttribute(attribute) for attribute in dataset.ATTRIBUTES]
        )

        user_attribute_name2attribute_name = {
            attr_name: attr_name for attr_name in dataset.ATTRIBUTES
        }

        # preprocess the data
        preprocessing_phase = ASETPipeline([
            StanzaNERExtractor(),
            SpacyNERExtractor("SpacyEnCoreWebLg"),
            ContextSentenceCacher(),
            CopyNormalizer(),
            OntoNotesLabelParaphraser(),
            SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
            FastTextLabelEmbedder("FastTextEmbedding", True, [" ", "_"]),
            SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
            BERTContextSentenceEmbedder("BertLargeCasedResource"),
            RelativePositionEmbedder()
        ])

        preprocessing_phase(
            document_base,
            EmptyInteractionCallback(),
            EmptyStatusCallback(),
            Statistics(do_collect=False)
        )

        # perform automatic matching
        random.seed(794009)

        matching_phase = ASETPipeline([
            SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
            ContextSentenceCacher(),
            FastTextLabelEmbedder("FastTextEmbedding", True, [" ", "_"]),
            RankingBasedMatcher(
                distance=SignalsMeanDistance(
                    signal_identifiers=[
                        "LabelEmbeddingSignal",
                        "TextEmbeddingSignal",
                        "ContextSentenceEmbeddingSignal",
                        "RelativePositionSignal"
                    ]
                ),
                max_num_feedback=20,
                len_ranked_list=10,
                max_distance=0.2,  # 0.35
                num_random_docs=1,
                sampling_mode="AT_MAX_DISTANCE_THRESHOLD",  # "MOST_UNCERTAIN_WITH_RANDOMS"
                adjust_threshold=True
            )
        ])

        matching_phase(
            document_base=document_base,
            interaction_callback=AutomaticRandomRankingBasedMatchingFeedback(
                documents,
                user_attribute_name2attribute_name
            ),
            status_callback=EmptyStatusCallback(),
            statistics=Statistics(do_collect=False)
        )

        ################################################################################################################
        # automatic execution of the interactive grouping based on the ground-truth information from T-REx
        ################################################################################################################
        documents = dataset.load_dataset()


        def get_uri_for_nugget(nugget):
            for doc in documents:
                if doc["id"] == nugget.document.name:
                    for mention in doc["mentions"][ATTRIBUTE_NAME]:
                        if consider_overlap_as_match(mention["start_char"], mention["end_char"],
                                                     nugget.start_char, nugget.end_char):
                            return mention["uri"]
            return "no-match"


        FINAL_CLUSTERS = None


        def merging_grouper_callback_fn(pipeline_element_identifier, data):
            global FINAL_CLUSTERS
            if pipeline_element_identifier == MergeGrouper.identifier:
                logger.info(f"Request: {data['request-name']}")
                if data["request-name"] == "get-attribute":
                    attribute = list(filter(lambda x: x.name == ATTRIBUTE_NAME, data["attributes"]))[0]
                    logger.info(f"Choose attribute '{attribute.name}'.")
                    return {
                        "attribute": attribute
                    }
                elif data["request-name"] == "output-clusters":
                    FINAL_CLUSTERS = data["clusters"]
                    return {}
                elif data["request-name"] == "same-cluster-feedback":
                    logger.info("Cluster 1:" + ", ".join(f"'{nugget.text}'" for nugget in data["cluster-1"]))
                    logger.info("Cluster 2:" + ", ".join(f"'{nugget.text}'" for nugget in data["cluster-2"]))
                    logger.info(f"Inter-cluster Distance: {data['inter-cluster-distance']}")

                    cluster_1_uri = get_uri_for_nugget(data["cluster-1"][0])
                    cluster_2_uri = get_uri_for_nugget(data["cluster-2"][0])

                    if cluster_1_uri is None or cluster_2_uri is None:
                        assert False, "Unknown cluster uris!"

                    # give feedback based on clusters' uris
                    if cluster_1_uri == cluster_2_uri:
                        logger.info("Feedback: SAME CLUSTER")
                        return {
                            "feedback": True
                        }
                    else:
                        logger.info("Feedback: NOT SAME CLUSTER")
                        return {
                            "feedback": False
                        }
                else:
                    assert False, "Unknown request!"
            else:
                assert False, "Unknown pipeline element!"


        from aset.querying.grouping import MergeGrouper

        pipeline = ASETPipeline(
            [
                MergeGrouper(
                    distance=SignalsMeanDistance(
                        signal_identifiers=[
                            "TextEmbeddingSignal",
                            "ContextSentenceEmbeddingSignal"
                        ]
                    ),
                    max_tries_no_merge=4,
                    skip=2,
                    automatically_merge_same_surface_form=True
                )
            ]
        )

        interaction_callback = InteractionCallback(merging_grouper_callback_fn)
        statistics = Statistics(do_collect=True)

        # execute the pipeline
        pipeline(
            document_base,
            interaction_callback,
            EmptyStatusCallback(),
            statistics
        )

        pred_clusters = list(FINAL_CLUSTERS.values())
        print("Number of predicted clusters:", len(pred_clusters))

        # get nuggets and corresponding uris
        all_nuggets = []
        for cluster in pred_clusters:
            all_nuggets += cluster
        print("Number of nuggets:", len(all_nuggets))

        all_uris = [get_uri_for_nugget(nugget) for nugget in all_nuggets]

        ################################################################################################################
        # get the ground-truth results
        ################################################################################################################

        # get ground-truth clustering
        all_mentions = []
        all_doc_ids = []
        for document in documents:
            if document["mentions"][ATTRIBUTE_NAME] != []:  # in case there is no mention ==> do not add to ground truth
                mention = document["mentions"][ATTRIBUTE_NAME][0]
                mention["text"] = document["text"][mention["start_char"]:mention["end_char"]]
                all_mentions.append(mention)
                all_doc_ids.append(document["id"])

        true_clusters = collections.defaultdict(list)
        true_clusters_doc_ids = collections.defaultdict(list)
        true_clusters_with_doc_ids = collections.defaultdict(list)
        for mention, doc_id in zip(all_mentions, all_doc_ids):
            true_clusters[mention["uri"]].append(mention["text"])
            true_clusters_doc_ids[mention["uri"]].append(doc_id)
            true_clusters_with_doc_ids[mention["uri"]].append((mention["text"], doc_id))

        true_clusters = list(true_clusters.values())
        true_clusters_doc_ids = list(true_clusters_doc_ids.values())

        true_clusters_and_doc_ids = list(zip(true_clusters, true_clusters_doc_ids))
        print("Number of true clusters:", len(true_clusters_and_doc_ids))

        ################################################################################################################
        # display the results
        ################################################################################################################

        true_clusters_and_doc_ids.sort(key=lambda x: len(x[0]), reverse=True)
        pred_clusters.sort(key=len, reverse=True)

        print("\n" * 4)
        print("COMPLETE RESULTS: (use with exp_1_wannadb_compute_scores.ipynb")

        print("True Clusters combined:")
        print(list(true_clusters_with_doc_ids.values()))

        print("\n" * 4)
        print("Predicted Clusters combined:")
        pred_clusters_with_doc_ids = []
        for cluster in pred_clusters:
            cluster_with_doc_ids = []
            for nugget in cluster:
                cluster_with_doc_ids.append((nugget.text, nugget.document.name))
            pred_clusters_with_doc_ids.append(cluster_with_doc_ids)
        print(pred_clusters_with_doc_ids)

        print("\n" * 4)
        print("HUMAN-READABLE RESULTS:")

        print("True Clusters:")
        for ix, (mentions, doc_ids) in enumerate(true_clusters_and_doc_ids):
            counter = collections.Counter(mention for mention in mentions)
            print(f"{ix + 1}.".rjust(4), ", ".join(f"'{text}' x {count}" for text, count in counter.most_common()),
                  f"({len(mentions)} total)")

        print("\n" * 4)
        print("Predicted Clusters:")
        for ix, cluster in enumerate(pred_clusters):
            counter = collections.Counter(nugget.text for nugget in cluster)
            print(f"{ix + 1}.".rjust(4), ", ".join(f"'{text}' x {count}" for text, count in counter.most_common()),
                  f"({len(cluster)} total)")
