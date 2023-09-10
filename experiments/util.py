import pandas as pd

from wannadb.data.data import DocumentBase
from wannadb.data.signals import NaturalLanguageLabelSignal, LabelSignal, CachedContextSentenceSignal


def consider_overlap_as_match(true_start, true_end, pred_start, pred_end):
    """Determines whether the predicted span is considered a match of the true span."""
    # considered as overlap if at least half of the larger span
    pred_length = pred_end - pred_start
    true_length = true_end - true_start

    valid_overlap = max(pred_length // 2, true_length // 2, 1)

    if pred_start <= true_start:
        actual_overlap = min(pred_end - true_start, true_length)
    else:
        actual_overlap = min(true_end - pred_start, pred_length)

    return actual_overlap >= valid_overlap


def create_dataframes_attributes_nuggets(document_base: DocumentBase):
    for document in document_base.documents:
        attributes_and_matches_df = pd.DataFrame({
            "attribute": document_base.attributes,  # object ==> cannot be written to csv
            "raw_attribute_name": [attribute.name for attribute in document_base.attributes],
            "nl_attribute_name": [attribute[NaturalLanguageLabelSignal] for attribute in document_base.attributes],
            "matching_nuggets": [document.attribute_mappings[attribute.name] for attribute in
                                    document_base.attributes],  # objects ==> cannot be written to csv
            "matching_nugget_texts": [[n.text for n in document.attribute_mappings[attribute.name]] for attribute in
                                        document_base.attributes]
        })

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        #print(attributes_and_matches_df)

        nuggets_df = pd.DataFrame({
            "nugget": document.nuggets,  # object ==> cannot be written to csv
            "raw_nugget_label": [nugget[LabelSignal] for nugget in document.nuggets],
            "nl_nugget_label": [nugget[NaturalLanguageLabelSignal] for nugget in document.nuggets],
            "nugget_text": [nugget.text for nugget in document.nuggets],
            "context_sentence": [nugget[CachedContextSentenceSignal]["text"] for nugget in document.nuggets],
            "start_char_in_context": [nugget[CachedContextSentenceSignal]["start_char"] for nugget in
                                        document.nuggets],
            "end_char_in_context": [nugget[CachedContextSentenceSignal]["end_char"] for nugget in document.nuggets]
        })

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        #print(nuggets_df)
    return attributes_and_matches_df, nuggets_df