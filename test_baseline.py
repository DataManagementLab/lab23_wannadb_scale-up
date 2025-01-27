import logging.config
import os
import timeit

from wannadb.configuration import BasePipelineElement, Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTLabelEmbedder, SBERTTextEmbedder, BERTContextSentenceEmbedder, \
    RelativePositionEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback

input_path = "/data/corona/raw-documents"


def write_results(res: str):
    with open("results.txt", "a") as file:
        file.write(res)

def start():
    with ResourceManager():
        logger = logging.getLogger()

        documents = []
        for filename in os.listdir(input_path):
            with open(os.path.join(input_path, filename), "r", encoding='utf-8') as infile:
                text = infile.read()
                documents.append(Document(filename.split(".")[0], text))

        logger.info(f"Loaded {len(documents)} documents")
        pipeline_elements: [BasePipelineElement] = [
            StanzaNERExtractor(),
            SpacyNERExtractor("SpacyEnCoreWebLg"),
            ContextSentenceCacher(),
            CopyNormalizer(),
            OntoNotesLabelParaphraser(),
            # SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
            SBERTLabelEmbedder("SBERTBertLargeNliMeanTokensResource"),
            SBERTTextEmbedder("SBERTBertLargeNliMeanTokensResource"),
            BERTContextSentenceEmbedder("BertLargeCasedResource"),
            RelativePositionEmbedder()
        ]

        wannadb_pipeline = Pipeline(pipeline_elements)

        document_base = DocumentBase(documents, [])

        statistics = Statistics(do_collect=True)
        statistics["preprocessing"]["config"] = wannadb_pipeline.to_config()
        start = timeit.default_timer()
        wannadb_pipeline(
            document_base=document_base,
            interaction_callback=EmptyInteractionCallback(),
            status_callback=EmptyStatusCallback(),
            statistics=statistics["preprocessing"]
        )
        stop = timeit.default_timer()

        output = "Baseline: {}".format((stop - start))

if __name__ == "__main__":
    start()