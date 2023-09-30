import logging.config
import os

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

input_path = "C:\\Users\\waq_5\\Documents\\GitHub\\lab23_wannadb_scale-up_main\\data\\corona\\raw-documents"

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

        wannadb_pipeline(
            document_base=document_base,
            interaction_callback=EmptyInteractionCallback(),
            status_callback=EmptyStatusCallback(),
            statistics=statistics["preprocessing"]
        )

if __name__ == "__main__":
    start()