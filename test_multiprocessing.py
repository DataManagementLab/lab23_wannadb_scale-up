import multiprocessing
import os
from typing import Any

from wannadb.configuration import BasePipelineElement, Pipeline
from wannadb.data.data import DocumentBase, Document
from wannadb.preprocessing.embedding import SBERTLabelEmbedder, SBERTTextEmbedder, BERTContextSentenceEmbedder, \
    RelativePositionEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager

NUM_PIPELINES = 3

def load_documents(input_path: str) -> [Document]:
    documents = []
    for filename in os.listdir(input_path):
        with open(os.path.join(input_path, filename), "r", encoding='utf-8') as infile:
            text = infile.read()
            documents.append(Document(filename.split(".")[0], text))
    return documents

def split_list(lst: [Any], n: int):
    if n <= 0:
        raise ValueError("Number of sublists (n) must be greater than 0")
    sublist_size = len(lst) // n
    remainder = len(lst) % n
    sublists = [lst[i * sublist_size:(i + 1) * sublist_size] for i in range(n)]

    # Distribute any remaining elements evenly among sublists
    for i in range(remainder):
        sublists[i].append(lst[n * sublist_size + i])

    return sublists

input_path = "./data/corona/raw-documents"
attributes = ["test, attribute"]

def startWannaDb():
    with multiprocessing.Manager() as manager:
        document_list = split_list(load_documents(input_path), NUM_PIPELINES)

        nugget_holder = manager.dict()
        document_bases: [DocumentBase] = [DocumentBase(document_list[ix], attributes) for ix in range(NUM_PIPELINES)]

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

        pipelines: [Pipeline] = [Pipeline(pipeline_elements) for _ in range(0, NUM_PIPELINES)]

        processes = [multiprocessing.Process(target=pipelines[ix].__call__, args=(document_bases[ix], nugget_holder)) for ix in range(len(pipelines))]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

if __name__ == "__main__":
    with ResourceManager() as resource_manager:
        multiprocessing.set_start_method("spawn")
        startWannaDb()
