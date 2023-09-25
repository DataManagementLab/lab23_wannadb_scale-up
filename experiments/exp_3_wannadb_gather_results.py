import collections
import json
import logging

import pandas as pd

import datasets.aviation.aviation as dataset  # TODO: choose data set by changing 'aviation' to 'corona', 'countries', 'nobel', or 'skyscrapers'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

RESULTS_DIRECTORY_PATH = f"results/{dataset.NAME}/feedback/"
RESULTS_FILENAME = "exp-3.json"
OUTPUT_PATH = f"results/{dataset.NAME}/feedback/f1_scores.csv"
START = 1
STOP = 40
STEP = 3

if __name__ == "__main__":
    output = collections.defaultdict(list)
    for num_feedback in list(range(START, STOP + 1, STEP)):
        path = f"{RESULTS_DIRECTORY_PATH}{num_feedback}/{RESULTS_FILENAME}"
        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)
        for attribute in dataset.ATTRIBUTES:
            output[attribute].append(results["matching"]["results"][attribute]["f1_score"])
        output["macro-f1"].append(results["matching"]["results"]["final_macro_f1"])

    df = pd.DataFrame(output)
    df.to_csv(OUTPUT_PATH)
