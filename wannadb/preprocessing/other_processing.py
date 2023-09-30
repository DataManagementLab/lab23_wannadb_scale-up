import logging
import os
from typing import Dict, List, Any

from wannadb.configuration import BasePipelineElement, register_configurable_element
from wannadb.data.data import DocumentBase, InformationNugget
from wannadb.data.signals import CachedContextSentenceSignal, \
    SentenceStartCharsSignal
from wannadb.interaction import BaseInteractionCallback
from wannadb.statistics import Statistics
from wannadb.status import BaseStatusCallback

logger: logging.Logger = logging.getLogger(__name__)


@register_configurable_element
class ContextSentenceCacher(BasePipelineElement):
    """Caches a nugget's context sentence."""

    identifier: str = "ContextSentenceCacher"

    required_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [],
        "attributes": [],
        "documents": [SentenceStartCharsSignal.identifier]
    }

    generated_signal_identifiers: Dict[str, List[str]] = {
        "nuggets": [CachedContextSentenceSignal.identifier],
        "attributes": [],
        "documents": []
    }

    def __init__(self):
        """Initialize the ContextSentenceCacher."""
        super(ContextSentenceCacher, self).__init__()
        print(f"Initialized '{self.identifier}'.")

    def __call__(
            self,
            nugget_holder
    ) -> None:
        while True:
            document = self.input_queue.get()
            if document is None:
                if self.next_pipeline_element is not None:
                    self.next_pipeline_element.input_queue.put(None)
                break
            print("ContextSentenceCacher: DOCUMENT:{} --- PIPELINE_PID:{}".format(document.name, os.getppid()))
            nuggets: [InformationNugget] = nugget_holder[document.name]
            for nugget in nuggets:
                self._call(nugget)
            nugget_holder[document.name] = nuggets  # Overwrite current dir entry

            if self.next_pipeline_element is not None:
                self.next_pipeline_element.input_queue.put(document)

    def _call(self, nugget: InformationNugget) -> None:
        sent_start_chars: List[int] = nugget.document[SentenceStartCharsSignal]
        context_start_char: int = 0
        context_end_char: int = 0
        for ix, sent_start_char in enumerate(sent_start_chars):
            if sent_start_char > nugget.start_char:
                if ix == 0:
                    context_start_char: int = 0
                    context_end_char: int = sent_start_char
                    break
                else:
                    context_start_char: int = sent_start_chars[ix - 1]
                    context_end_char: int = sent_start_char
                    break
        else:
            if sent_start_chars != []:
                context_start_char: int = sent_start_chars[-1]
                context_end_char: int = len(nugget.document.text)

        context_sentence: str = nugget.document.text[context_start_char:context_end_char]
        start_in_context: int = nugget.start_char - context_start_char
        end_in_context: int = nugget.end_char - context_start_char

        nugget[CachedContextSentenceSignal] = CachedContextSentenceSignal({
            "text": context_sentence,
            "start_char": start_in_context,
            "end_char": end_in_context
        })

    def to_config(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ContextSentenceCacher":
        return cls()
