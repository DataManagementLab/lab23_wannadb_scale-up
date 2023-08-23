import os
import logging

from wannadb.statistics import Statistics
from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.extraction import StanzaNERExtractor
from wannadb.resources import ResourceManager
from wannadb.status import EmptyStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


def run_tests():
    logger.debug("Started exttr")
    input_path = "/home/rami/lab23_wannadb_scale-up/data/corona/raw-documents"
    with ResourceManager():
        documents = []
        for filename in os.listdir(input_path):
            with open(os.path.join(input_path, filename), "r", encoding='utf-8') as infile:
                text = infile.read()
                documents.append(Document(filename.split(".")[0], text))

        logger.info(f"Loaded {len(documents)} documents")
        statistics = Statistics(do_collect=True)

        wannadb_pipeline = Pipeline([
            StanzaNERExtractor(),
        ])

        document_base = DocumentBase(documents, [])

        wannadb_pipeline(
            document_base=document_base,
            interaction_callback=EmptyInteractionCallback(),
            status_callback=EmptyStatusCallback(),
            statistics=statistics["preprocessing"]
        )


if __name__ == "__main__":
    run_tests()