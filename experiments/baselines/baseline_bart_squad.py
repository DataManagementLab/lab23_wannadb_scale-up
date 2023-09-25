import os
import json
import numpy as np
from tqdm import tqdm
from docopt import docopt
from pathlib import Path
from transformers.pipelines import pipeline

from aset.statistics import Statistics

usage = """
Usage:
    baseline_bart_squad.py <dataset> [-v]

Options:
    <dataset>   The dataset to use, one of: aviation, corona, countries, nobel, skyscrapers
    -v          Verbose, prints results per attributes

"""
def calculate_f1_scores(results: Statistics):
    """
    compute the evaluation metrics per attribute
    """

    # recall
    if (results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"] + results["num_should_be_filled_is_empty"]) == 0:
        results["recall"] = 1
    else:
        results["recall"] = results["num_should_be_filled_is_correct"] / (
                results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"] +
                results["num_should_be_filled_is_empty"])

    # precision
    if (results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"] + results["num_should_be_empty_is_full"]) == 0:
        results["precision"] = 1
    else:
        results["precision"] = results["num_should_be_filled_is_correct"] / (
                results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"] + results["num_should_be_empty_is_full"])

    # f1 score
    if results["precision"] + results["recall"] == 0:
        results["f1_score"] = 0
    else:
        results["f1_score"] = (
                2 * results["precision"] * results["recall"] / (results["precision"] + results["recall"]))


def run_evaluation(input_files, squad_model_pipeline):
    statistics = Statistics(do_collect = True)
    for filename in tqdm(input_files):    
        # load datapoint
        with open(filename, "r", encoding="utf8") as file:
            datapoint = json.load(file)
            
        attribute = datapoint["attribute"]
        
        # initialize statistics
        if not datapoint["attribute"] in statistics.all_keys():  
            statistics[attribute]["num_should_be_filled_is_empty"] = 0
            statistics[attribute]["num_should_be_filled_is_correct"] = 0
            statistics[attribute]["num_should_be_filled_is_incorrect"] = 0
            statistics[attribute]["num_should_be_empty_is_empty"] = 0
            statistics[attribute]["num_should_be_empty_is_full"] = 0
            
        # prepare input format
        inputs = { 'question': datapoint["question"],
                'context': datapoint["context"]}

        # run model on input
        output = squad_model_pipeline(inputs)

        # only exact matches are counted as correct
        if output["answer"] == datapoint["label"]:
            statistics[attribute]["num_should_be_filled_is_correct"] += 1
        elif output["answer"][:-1] == datapoint["label"]:
            statistics[attribute]["num_should_be_filled_is_correct"] += 1
        else:
            # value can be supposed to be empty
            if datapoint["label"] == "":
                if output["answer"] != "":
                    statistics[attribute]["num_should_be_empty_is_full"] += 1
            else:
                statistics[attribute]["num_should_be_filled_is_incorrect"] += 1

    return statistics
        

if __name__ == "__main__":
    args = docopt(usage)

    dataset = args["<dataset>"]
    
    # load model checkpoint
    model_name = "phiyodr/bart-large-finetuned-squad2"
    squad_model_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)
    print(f"Loaded *{model_name}* model checkpoint")

    print(f"Starting evaluation of BART trained on SQuAD 2.0 on *{dataset}* data")


    test_data_file = Path(f"datasets/{dataset}/bart_squad_qa_data_test_files.json")
    if not test_data_file.exists():
        print("Data needs to brought into the correct format first, please run 'bart_baselines_create_datasets.py' script.")
    with open(test_data_file, "r", encoding="utf8") as file: 
        input_files = json.load(file)
            
    print(f"Have {len(input_files)} test datapoints")

    # run evaluation on the test data
    statistics = run_evaluation(input_files, squad_model_pipeline)

    # compute F1 scores
    dataset_overall_f1_score = []
    for attribute in statistics.all_keys():
        calculate_f1_scores(statistics[attribute])
        dataset_overall_f1_score.append(statistics[attribute]["f1_score"])
    statistics["macro-f1"] = np.mean(dataset_overall_f1_score)
    statistics["eval_data"] = str(test_data_file)
    statistics["model_name"] = model_name
    print("----------------------------------------------")
    print(f"Overall F1 score on the *{dataset}* dataset:", np.mean(dataset_overall_f1_score))

    # saving evaluation results as json file
    save_path = Path(f"results_SQuAD_{dataset}.json")
    with open(save_path, "w") as file:
        json.dump(statistics.to_serializable(), file, indent=5)
    print("----------------------------------------------")
    print("Saved results: ", save_path)

    # print results per attribute:
    if args["-v"]:
        for attr in statistics.all_keys():
            print(attr, statistics[attr]["f1_score"])
