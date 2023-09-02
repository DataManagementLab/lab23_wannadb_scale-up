import os
import logging
from glob import glob
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

NAME: str = "wikipedia"

# Sie können später zusätzliche Attribute hinzufügen, aber im Moment haben wir nur 'text'.
ATTRIBUTES: List[str] = [
    "text",  # text content
]

BASE_PATH = r"C:\Users\Pascal\Desktop\WannaDB\lab23_wannadb_scale-up\datasets\wikipedia"


def load_dataset(subset_name: str) -> List[Dict[str, Any]]:
    """
    Load the wikipedia dataset for a specific subset.

    This method requires the .txt files in the "C:\Users\Pascal\Desktop\WannaDB\lab23_wannadb_scale-up\datasets\wikipedia\<subset_name>\" folder.
    """
    dataset: List[Dict[str, Any]] = []
    path: str = os.path.join(BASE_PATH, subset_name, "*.txt")
    for file_path in glob(path):
        with open(file_path, encoding="utf-8") as file:
            document = {
                "id": os.path.basename(file_path).replace(".txt", ""),  # Using the filename as ID
                "text": file.read()
            }
            dataset.append(document)
    return dataset


def write_document(subset_name: str, document: Dict[str, Any]) -> None:
    """
    Write the given document to the dataset.
    """
    path: str = os.path.join(BASE_PATH, subset_name, document["id"] + ".txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(document["text"])