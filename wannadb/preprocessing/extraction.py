import abc
import json
import logging
import os
from typing import Any, Dict, List

import requests
import spacy
import stanza
from spacy import Language
from spacy.tokens import Doc

from wannadb import resources
from wannadb.configuration import register_configurable_element, BasePipelineElement
from wannadb.data.data import DocumentBase, InformationNugget, Document
from wannadb.data.signals import LabelSignal, POSTagsSignal, SentenceStartCharsSignal
from wannadb.interaction import BaseInteractionCallback
from wannadb.resources import StanzaNERPipeline, FigerNERPipeline
from wannadb.statistics import Statistics
from wannadb.status import BaseStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


class BaseExtractor(BasePipelineElement, abc.ABC):
    """
    Base class for all extractors.

    Extractors derive the information nuggets from the documents.
    """
    identifier: str = "BaseExtractor"


########################################################################################################################
# actual extractors
########################################################################################################################


@register_configurable_element
class SpacyNERExtractor(BaseExtractor):
    """Extractor based on spacy's NER models."""

    identifier: str = "SpacyNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self, spacy_resource_identifier: str) -> None:
        """
        Initialize the SpacyNERExtractor.

        :param spacy_resource_identifier: identifier of the spacy model resource
        """
        super(SpacyNERExtractor, self).__init__()
        self._spacy_resource_identifier: str = spacy_resource_identifier

        # preload required resources
        self._spacy_nlp: Language = spacy.load("en_core_web_lg")
        print(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document: Document
    ) -> [InformationNugget]:
        print("Spacy_NER: DOCUMENT:{} --- PIPELINE_PID:{}".format(document.name, os.getppid()))
        spacy_output: Doc = self._spacy_nlp(document.text)
        sentence_start_chars: List[int] = []

        # transform the spacy output into the document and nuggets
        for sentence in spacy_output.sents:
            sentence_start_chars.append(sentence.start_char)

        document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)

        nuggets: [InformationNugget] = []
        for entity in spacy_output.ents:
            nugget: InformationNugget = InformationNugget(
                document=document,
                start_char=entity.start_char,
                end_char=entity.end_char
            )

            nugget[POSTagsSignal] = POSTagsSignal([])  # TODO: gather pos tags
            nugget[LabelSignal] = LabelSignal(entity.label_)

            nuggets.append(nugget)
        return nuggets

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "spacy_resource_identifier": self._spacy_resource_identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpacyNERExtractor":
        return cls(config["spacy_resource_identifier"])


@register_configurable_element
class StanzaNERExtractor(BaseExtractor):
    """Extractor based on Stanza's NER model."""

    identifier: str = "StanzaNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier, POSTagsSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self) -> None:
        """Initialize the StanzaNERExtractor."""
        super(StanzaNERExtractor, self).__init__()

        # preload required resources
        path: str = os.path.join(os.path.dirname(__file__), "../..", "models", "stanza")
        self._nlp = stanza.Pipeline(
            lang="en", processors="tokenize,mwt,pos,ner", model_dir=path, verbose=False
        )
        print(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document: Document
    ) -> [InformationNugget]:
        print("Stanza_NER: DOCUMENT:{} --- PIPELINE_PID:{}".format(document.name, os.getppid()))
        stanza_output = self._nlp(document.text)

        sentence_start_chars: List[int] = []

        # transform the stanza output into the document and nuggets
        nuggets: [InformationNugget] = []
        for sentence in stanza_output.sentences:
            sentence_start_chars.append(sentence.tokens[0].start_char)

            for entity in sentence.entities:
                nugget: InformationNugget = InformationNugget(
                    document=document,
                    start_char=entity.start_char,
                    end_char=entity.start_char + len(entity.text)
                )

                nugget[POSTagsSignal] = POSTagsSignal([word.xpos for word in entity.words])
                nugget[LabelSignal] = LabelSignal(entity.type)

                nuggets.append(nugget)

        document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)
        return nuggets

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StanzaNERExtractor":
        return cls()


@register_configurable_element
class FigerNERExtractor(BaseExtractor):
    """
    Extractor based on Figer's NER model
    (using CoreNLP for basic extraction and fine-graned labeling on top).
    """

    identifier: str = "FigerNERExtractor"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": []
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [LabelSignal.identifier],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    def __init__(self) -> None:
        """Initialize the FigerNERExtractor."""
        super().__init__()

        # preload required resources
        resources.MANAGER.load(FigerNERPipeline)
        logger.debug(f"Initialized '{self.identifier}'.")

    def _call(
            self,
            document_base: DocumentBase,
            interaction_callback: BaseInteractionCallback,
            status_callback: BaseStatusCallback,
            statistics: Statistics
    ) -> None:
        statistics["num_documents"] = len(document_base.documents)

        base_url = resources.MANAGER[FigerNERPipeline]

        for ix, document in enumerate(document_base.documents):
            self._use_status_callback(status_callback, ix, len(document_base.documents))

            # Run FIGER on document (truncate to first 8000 chars if necessary due to server limitations)
            r = requests.get(base_url, params={'text': document.text[:8000]})
            if r.status_code == 200:
                answer = json.loads(r.text)
                if answer["status"] == 200:
                    sentence_start_chars: List[int] = answer["sentence_offsets"]

                    for raw_nugget in answer["data"]:
                        nugget: InformationNugget = InformationNugget(
                            document=document,
                            start_char=raw_nugget["start_char"],
                            end_char=raw_nugget["end_char"]
                        )

                        # nugget[POSTagsSignal] = POSTagsSignal([word.xpos for word in entity.words])

                        # Label format from FIGER is e.g.
                        # "/location@1.4898770776826524,/organization/company@0.17639383484191654,/location/country@0.25034040521054085"
                        # Extract first label (without numeric value)
                        label_string = raw_nugget['label'].split(',')[0].split('@')[0][1:].replace("/", " ")
                        nugget[LabelSignal] = LabelSignal(label_string)
                        document.nuggets.append(nugget)

                        statistics["num_nuggets"] += 1
                        statistics["figer_label_dist"][label_string] += 1

                    document[SentenceStartCharsSignal] = SentenceStartCharsSignal(sentence_start_chars)
                else:
                    print(logger.warning(f"Failed to run FIGER on document '{document.name}' with error '{answer['error']}'"))
            else:
                logger.warning(f"Failed to run FIGER on document '{document.name}'")

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FigerNERExtractor":
        return cls()
