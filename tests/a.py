import abc
import logging
import random
import time

from typing import Any, Dict, List, Callable, Tuple

import numpy as np

from wannadb.configuration import BasePipelineElement, register_configurable_element, Pipeline

from wannadb.data.data import Document, DocumentBase, InformationNugget
from wannadb.data.signals import CachedContextSentenceSignal, CachedDistanceSignal, \
    SentenceStartCharsSignal, CurrentMatchIndexSignal, LabelSignal
from wannadb.interaction import BaseInteractionCallback

from wannadb.matching.distance import BaseDistance

from wannadb.statistics import Statistics
from wannadb.status import BaseStatusCallback

import cProfile
import io
import os
from pstats import SortKey
import pstats
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from typing import List, Any, Tuple
from wannadb.data.data import DocumentBase
from typing import List, Any, Optional, Union
import re
import logging
from wannadb.data.data import DocumentBase, Attribute, Document, InformationNugget
import time
import numpy as np
from wannadb.statistics import Statistics
from wannadb.data.signals import LabelEmbeddingSignal, TextEmbeddingSignal, ContextSentenceEmbeddingSignal, RelativePositionSignal,UserProvidedExamplesSignal, CachedDistanceSignal, CurrentMatchIndexSignal, NewCurrentMatchIndexSignal
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances


from wannadb.configuration import Pipeline
from wannadb.data.data import Document, DocumentBase
from wannadb.interaction import EmptyInteractionCallback
from wannadb.preprocessing.embedding import SBERTTextEmbedder, SBERTExamplesEmbedder
from wannadb.preprocessing.extraction import StanzaNERExtractor, SpacyNERExtractor
from wannadb.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from wannadb.preprocessing.normalization import CopyNormalizer
from wannadb.preprocessing.other_processing import ContextSentenceCacher
from wannadb.resources import ResourceManager
from wannadb.statistics import Statistics
from wannadb.status import EmptyStatusCallback

import cProfile, pstats, io
from pstats import SortKey
