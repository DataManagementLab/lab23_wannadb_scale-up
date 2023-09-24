"""
wikipedia tiny Dataset
==============
"""

import json
import logging
import os
from glob import glob
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

NAME: str = "wikipedia_tiny"

ATTRIBUTES: List[str] = [
    "country",
    "date of birth",
    "city"
]


def load_dataset() -> List[Dict[str, Any]]:
    """
    Load the wikipedia tiny dataset.

    This method requires the .txt files in the "datasets/wikipedia/tiny/documents/" folder.
    """
    dataset: List[Dict[str, Any]] = []
    path: str = os.path.join(os.path.dirname(__file__), "documents", "*.txt")
    for file_path in glob(path):
        with open(file_path, encoding="utf-8") as file:
            dataset.append(file.read())
    return dataset


def write_document(document: Dict[str, Any]) -> Any:
    """
    Write the given document to the dataset.
    """
    path: str = os.path.join(os.path.dirname(__file__), "documents", str(document["id"]) + ".txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(document))
