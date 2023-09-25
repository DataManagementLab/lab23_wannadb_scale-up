import os
import logging
from glob import glob
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

NAME: str = "wikipedia"


ATTRIBUTES: List[str] = [
    "name",  
    "birth date",
    "country", #tbd
]



def load_dataset(subset_name: str) -> List[Dict[str, Any]]:
    """
    Load the wikipedia dataset for a specific subset.

    This method requires the .json files in the "datasets/wikipedia/{subset}/documents/" folder.
    """
    dataset: List[Dict[str, Any]] = []
    path: str = os.path.join(os.path.dirname(__file__), subset_name, "*.txt")
    for file_path in glob(path):
        with open(file_path, encoding="utf-8") as file:
            document = {
                "id": os.path.basename(file_path).replace(".txt", ""),  
                "text": file.read()
            }
            dataset.append(document)
    return dataset


def write_document(subset_name: str, document: Dict[str, Any]) -> None:
    """
    Write the given document to the dataset.
    """
    path: str = os.path.join(os.path.dirname(__file__), subset_name, document["id"] + ".txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(document["text"])