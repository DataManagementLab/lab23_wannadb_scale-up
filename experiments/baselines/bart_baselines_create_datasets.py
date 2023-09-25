import json
import logging
import os
from pathlib import Path

from aset.configuration import ASETPipeline
from aset.data.data import ASETDocumentBase, ASETDocument, ASETAttribute
from aset.interaction import EmptyInteractionCallback
from aset.preprocessing.extraction import StanzaNERExtractor
from aset.preprocessing.label_paraphrasing import OntoNotesLabelParaphraser, SplitAttributeNameLabelParaphraser
from aset.preprocessing.normalization import CopyNormalizer
from aset.preprocessing.other_processing import ContextSentenceCacher
from aset.resources import ResourceManager
from aset.statistics import Statistics
from aset.status import EmptyStatusCallback
from experiments.util import consider_overlap_as_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

######################################################
################### Set Parameters here: #############
######################################################

# set this to the dataset to process ('aviation', 'corona', 'skyscrapers', 'countries' or 'nobel')
from datasets.countries import countries as dataset

# set to "squad" or "seq2seq"
baseline = "squad" 

######################################################
######################################################
######################################################

def create_dataset_single_attribute(aset_documents: list, dataset_documents: list, attributes: list, save_path: Path, val_split: float=0.15, test_split: float=0.1):
    '''
    From documents and attributes, create datapoints for bart seq2seq task and save them into datasets folder
    '''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    train_files, val_files, test_files = [], [], []
    num_val = int(val_split * len(aset_documents))
    num_test = int(test_split * len(aset_documents))
    print("Files", len(aset_documents), "val", num_val, "test", num_test)
    for i, aset_document in enumerate(aset_documents):
        dataset_document = [doc for doc in dataset_documents if doc["id"] == aset_document.name][0]
        document_text = aset_document.text
        for j, attr in enumerate(attributes):
            attr_text = attr.name
            datapoint_input = attr_text + "</s>" + "<mask>" + "</s>" + document_text
            # prepend an example for one-shot learning
            try:
                ground_truth = dataset_document["text"][dataset_document["mentions"][attr_text][0]["start_char"] : dataset_document["mentions"][attr_text][0]["end_char"] ]
            except IndexError:
                ground_truth = ""
            datapoint_label = attr_text + "</s>" + ground_truth
            datapoint = {"input": datapoint_input, "label": datapoint_label}
            
            filename = Path("doc_" + str(i) + "_attr_" + str(j) + ".json")
            with open(save_path / filename, "w", encoding="utf8") as file:
                json.dump(datapoint, file, indent=3)

            if i < num_val:
                val_files.append(str(save_path / filename))
            elif i >= num_val and i < num_val + num_test:
                test_files.append(str(save_path / filename))
            else:
                train_files.append(str(save_path / filename))
    print("num train", len(train_files), "num_test", len(test_files), "num val", len(val_files))

    with open(save_path + "_train_files.json", "w", encoding="utf8") as file:
        json.dump(train_files, file, indent=3)
    with open(save_path + "_val_files.json", "w", encoding="utf8") as file:
        json.dump(val_files, file, indent=3)
    with open(save_path + "_test_files.json", "w", encoding="utf8") as file:
        json.dump(test_files, file, indent=3)

def create_dataset_question_answering(aset_documents: list, dataset_documents: list, attributes: list, save_path: Path, val_split: float=0.15, test_split: float=0.1, squad_format:bool=False):
    '''
    From documents and attributes, create datapoints for bart seq2seq task and save them into datasets folder
    '''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    train_files, val_files, test_files = [], [], []
    num_val = int(val_split * len(aset_documents))
    num_test = int(test_split * len(aset_documents))
    print("Files", len(aset_documents), "val", num_val, "test", num_test)
    for i, aset_document in enumerate(aset_documents):
        dataset_document = [doc for doc in dataset_documents if doc["id"] == aset_document.name][0]
        document_text = aset_document.text
        for j, attr in enumerate(attributes):
            attr_text = attr.name
            attr_text_processed = " ".join(attr_text.split("_"))
            print("Attribute is: ", attr_text_processed)
            question = "What is the " + attr_text_processed + "?"
            context = document_text
            try:
                ground_truth = dataset_document["text"][dataset_document["mentions"][attr_text][0]["start_char"] : dataset_document["mentions"][attr_text][0]["end_char"] ]
            except IndexError:
                ground_truth = ""
            if squad_format:
                datapoint = {"attribute": attr_text, "question": question, "context": context, "label": ground_truth}
            else:
                input_txt = question + " Answer:<mask> Context: " + context
                datapoint = {"attribute": attr_text, "input": input_txt, "label": ground_truth}
            
            filename = Path("doc_" + str(i) + "_attr_" + str(j) + ".json")
            with open(save_path / filename, "w", encoding="utf8") as file:
                json.dump(datapoint, file, indent=3)

            if i < num_val:
                val_files.append(str(save_path / filename))
            elif i >= num_val and i < num_val + num_test:
                test_files.append(str(save_path / filename))
            else:
                train_files.append(str(save_path / filename))
    print("num train", len(train_files), "num_test", len(test_files), "num val", len(val_files))

    with open(save_path + "_train_files.json", "w", encoding="utf8") as file:
        json.dump(train_files, file, indent=3)
    with open(save_path + "_val_files.json", "w", encoding="utf8") as file:
        json.dump(val_files, file, indent=3)
    with open(save_path + "_test_files.json", "w", encoding="utf8") as file:
        json.dump(test_files, file, indent=3)

if __name__ == "__main__":
    """
    Bring the dataset in the correct format for the baseline

    dataset: either 'corona', 'aviation' or 'trex'
    baseline: either 'squad' or 'seq2seq'
    """
    with ResourceManager() as resource_manager:

        ################################################################################################################
        # create ground-truth document base (if not already cached)
        ################################################################################################################
        path = os.path.join(os.path.dirname(__file__), "../..", "cache", f"{dataset.NAME}-stanza-ground-truth-db.bson")

        if not os.path.isfile(path):
            # load the data
            documents = dataset.load_dataset()
            document_base = ASETDocumentBase(
                documents=[ASETDocument(doc["id"], doc["text"]) for doc in documents],
                attributes=[ASETAttribute(attribute) for attribute in dataset.ATTRIBUTES]
            )

            # preprocess the data
            preprocessing_phase = ASETPipeline(
                [
                    StanzaNERExtractor(),
                    ContextSentenceCacher(),
                    CopyNormalizer(),
                    OntoNotesLabelParaphraser(),
                    SplitAttributeNameLabelParaphraser(do_lowercase=True, splitters=[" ", "_"]),
                ]
            )

            preprocessing_phase(
                document_base,
                EmptyInteractionCallback(),
                EmptyStatusCallback(),
                Statistics(do_collect=False)
            )

            for aset_attribute in document_base.attributes:
                attribute = aset_attribute.name

                for aset_document in document_base.documents:
                    document = [doc for doc in documents if doc["id"] == aset_document.name][0]

                    aset_document.attribute_mappings[aset_attribute.name] = []

                    # find all valid matches
                    for aset_nugget in aset_document.nuggets:
                        for mention in document["mentions"][attribute]:
                            if consider_overlap_as_match(mention["start_char"], mention["end_char"],
                                                         aset_nugget.start_char,
                                                         aset_nugget.end_char, ):
                                aset_document.attribute_mappings[aset_attribute.name].append(aset_nugget)
                                break

            # save the document base
            with open(path, "wb") as file:
                file.write(document_base.to_bson())

        else:
            print("Have existing document collection")
            with open(path, "rb") as file:
                document_base = ASETDocumentBase.from_bson(file.read())

            # cached context sentences are not serialized because they are redundant ==> generate them again
            preprocessing_phase = ASETPipeline(
                [
                    ContextSentenceCacher()
                ]
            )

            preprocessing_phase(
                document_base,
                EmptyInteractionCallback(),
                EmptyStatusCallback(),
                Statistics(do_collect=False)
            )

        ################################################################################################################
        # actual experiments
        ################################################################################################################

        dataset_documents = dataset.load_dataset()

        if baseline == 'squad': 
            create_dataset_question_answering(document_base.documents, dataset_documents, document_base.attributes, f"datasets/{dataset.NAME}/bart_squad_qa_data", squad_format=True)
        
        if baseline == 'seq2seq':
            create_dataset_single_attribute(document_base.documents, dataset_documents, document_base.attributes, f"datasets/{dataset.NAME}/bart_seq2seq_data")
        print("Done creating the dataset files")


