import json
import os
import random
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_scheduler

from aset.statistics import Statistics


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name: ", torch.cuda.get_device_name(0))

    else: 
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    
    return device

class TextToAttributeDataset(Dataset):

    def __init__(self, filenames: list, tokenizer, use_labels:bool=True, generate_decoder_ids:bool=True):
        self.tokenizer = tokenizer
        self.max_length = 1024
        self.use_labels = use_labels
        self.generate_decoder_ids=generate_decoder_ids

        self.all_files = filenames
        self.total_files = len(filenames)

    def __len__(self):
        return self.total_files

    def __getitem__(self, index: int):
        chosen_datapoint_file = Path(self.all_files[index])

        with open(chosen_datapoint_file, "r", encoding="utf8") as file:
            chosen_datapoint = json.load(file)

        input_text = chosen_datapoint["input"]
        label_text = chosen_datapoint["label"]

        input_tokenized = self.tokenizer.encode_plus(input_text, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        #detokenized = self.tokenizer.convert_ids_to_tokens(input_tokenized["input_ids"])
        input_tokenized["input_str"] = input_text
        datapoint = input_tokenized
        
        if self.use_labels:
            label_tokenized = self.tokenizer.encode_plus(label_text, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            #detokenized = self.tokenizer.convert_ids_to_tokens(label_tokenized["input_ids"])
            label_tokenized["input_str"] = input_text
            pad_token_broadcasted=torch.full((1,1),self.tokenizer.pad_token_id)
            # label format is attr</s>value</s>
            datapoint["labels"] = torch.cat([label_tokenized["input_ids"][:, 1:], pad_token_broadcasted], dim=1)
            datapoint["labels_str"] = label_text
            
            # train decoder input format is <s>attr</s>value</s>
            decoder_input_text = label_text
            decoder_input = self.tokenizer.encode_plus(decoder_input_text, add_special_tokens=True, padding="max_length", max_length=self.max_length, return_tensors="pt", )
            datapoint["train_decoder_input_ids"] = decoder_input["input_ids"]
        if self.generate_decoder_ids:
            #eval decoder input format is <s>attr</s>
            decoder_input_text = label_text.split("</s>")[0]
            decoder_input = self.tokenizer.encode_plus(decoder_input_text, add_special_tokens=True, return_tensors="pt")
            datapoint["eval_decoder_input_ids"] = decoder_input["input_ids"]

            
        
        return datapoint       

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def generate_loop(model, inputs, decoder_inputs, max_length, end_token_id, attention_mask):
    prediction = decoder_inputs
    while True:
        model.eval()
        generated_output = model(inputs, decoder_input_ids = prediction, attention_mask=attention_mask)
        generated_ids = torch.argmax(generated_output["logits"], dim=-1)
        #(B, T)
        next_token = generated_ids[:, -1:]
        prediction=torch.cat([prediction, next_token], dim=1)
        if next_token[0,0]==end_token_id:
            break
        if prediction.shape[1]>=max_length:
            break
    return prediction

def calculate_f1_scores(results: Statistics):
    # compute the evaluation metrics per attribute

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

    # true negative rate
    if results["num_should_be_empty_is_empty"] + results["num_should_be_empty_is_full"] == 0:
        results["true_negative_rate"] = 1
    else:
        results["true_negative_rate"] = results["num_should_be_empty_is_empty"] / (results["num_should_be_empty_is_empty"] + results["num_should_be_empty_is_full"])

    # true positive rate
    if results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"] + results["num_should_be_filled_is_empty"] == 0:
        results["true_positive_rate"] = 1
    else:
        results["true_positive_rate"] = results["num_should_be_filled_is_correct"] / (results["num_should_be_filled_is_correct"] + results["num_should_be_filled_is_incorrect"] + results["num_should_be_filled_is_empty"])

def bart_evaluate(model:BartForConditionalGeneration, device, tokenizer, dataset, statistics:Statistics=None, verbose: bool=False, debug_mode:bool=False):
    # set model into evaluation mode
    model.eval()
    
    # to accumulate accuracy and loss over all datapoints
    accuracy_only_values = []

    if statistics == None:
        statistics = Statistics(do_collect = True)
    
    # randomly select 5 batches to print when verbose=True
    if verbose:
        num_to_print = 5
        print(f"Will randomly outprint {num_to_print} datapoints as examples")
        print_indices = random.sample(range(len(dataset)), min(num_to_print, len(dataset)))
    
    for datapoint_idx, datapoint in enumerate(dataset):
        t0 = time.time()
        with torch.no_grad():
            inputs = datapoint["input_ids"].squeeze(dim=1).long()
            decoder_inputs = datapoint["eval_decoder_input_ids"].squeeze(dim=1).long()
            attention_mask = datapoint["attention_mask"].squeeze(dim=1).long()
            #print(inputs.shape)
            inputs, decoder_inputs, attention_mask= inputs.to(device), decoder_inputs.to(device), attention_mask.to(device)
            #print("Generate inputs:")
            #print(bart_tokenizer.convert_ids_to_tokens(inputs[0], skip_special_tokens=False))

            generated_output = generate_loop(model, inputs,
                                            decoder_inputs = decoder_inputs,
                                            max_length=50,
                                            end_token_id=tokenizer.pad_token_id,
                                            attention_mask=attention_mask)

            #print("Size generated output:", generated_output.size())   
            #print(bart_tokenizer.convert_ids_to_tokens(generated_output[0], skip_special_tokens=False))       
            generated_output_decoded = bart_tokenizer.decode(generated_output[0], skip_special_tokens=False)

            
            if debug_mode:
                model.eval()
                debug_output = model(inputs, decoder_input_ids = decoder_inputs, attention_mask=attention_mask)
                predictions = torch.argmax(debug_output["logits"], dim=-1)
                print("="*100)
                print("Inputs")
                print(bart_tokenizer.convert_ids_to_tokens(inputs[0], skip_special_tokens=False))
                print("Labels")
                print(bart_tokenizer.convert_ids_to_tokens(datapoint["labels"][0], skip_special_tokens=False))
                print("Decoder Input")
                print(bart_tokenizer.convert_ids_to_tokens(decoder_inputs[0], skip_special_tokens=False))
                print("Output of model()")   
                print(bart_tokenizer.convert_ids_to_tokens(predictions[0], skip_special_tokens=False))
                print("Output generated:")
                print(bart_tokenizer.convert_ids_to_tokens(generated_output[0], skip_special_tokens=False))
            try:
                predicted_value = generated_output_decoded.split("</s>")[1]
                if predicted_value == "":
                    predicted_value = "empty"
            except:
                predicted_value = "empty"
            attribute = datapoint["labels_str"].split("</s>")[0]
            try:
                ground_truth_value = datapoint["labels_str"].split("</s>")[1]
                if ground_truth_value == "":
                    ground_truth_value = "empty"
            except:
                ground_truth_value = "empty"
            accuracy_only_values.append(predicted_value == ground_truth_value)

            if not attribute in statistics.all_keys():
                #print("Adding attribute: ", attribute)
                statistics[attribute]["num_should_be_filled_is_empty"] = 0
                statistics[attribute]["num_should_be_filled_is_correct"] = 0
                statistics[attribute]["num_should_be_filled_is_incorrect"] = 0
                statistics[attribute]["num_should_be_empty_is_empty"] = 0
                statistics[attribute]["num_should_be_empty_is_full"] = 0
            if ground_truth_value == "empty":
                if predicted_value == "empty":
                    statistics[attribute]["num_should_be_empty_is_empty"] += 1
                else:
                    statistics[attribute]["num_should_be_empty_is_full"] += 1
            else:
                if predicted_value == ground_truth_value:
                    statistics[attribute]["num_should_be_filled_is_correct"] += 1
                elif predicted_value == "empty":
                    statistics[attribute]["num_should_be_filled_is_empty"] += 1
                else:
                    statistics[attribute]["num_should_be_filled_is_incorrect"] += 1

        if verbose and datapoint_idx in print_indices:
            print("#### Datapoint: ")
            print("Attribute: ", attribute)
            print("Prediction:", predicted_value)
            print("Ground Truth:", ground_truth_value)
            print("Correct?", predicted_value == ground_truth_value)

    statistics[attribute]["accuracy_overall_values"] = np.mean(accuracy_only_values)

    evaluation_results = {
        "accuracy_only_values": np.mean(accuracy_only_values),
    }
    return evaluation_results


def train_BART_model(model, tokenizer, device, epochs, train_dataloader, val_dataset, train_dataset, optimizer, lr_scheduler, checkpoint_folder, debug_mode):
    best_val_accuracy = -1
    clip_value = 1.0

    checkpoint_folder_names = []
    loss_fn = torch.nn.CrossEntropyLoss()
    
        
    for epoch in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} || {'Train Loss':^12} | {'Train Acc Overall': ^17} | {'Train Acc Values': ^16} || {'Val Acc Values':^14} || {'Batch Time':^9}")
        print("-"*130)
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        
        data_iter = iter(train_dataloader)
        
        epoch_train_accuracy_only_value = []
        for step in range(len(train_dataloader)):
            # set model into training mode
            model.train()

            t0 = time.time()
            batch = next(data_iter)
            t1 = time.time()
            batch_counts +=1

            for param in model.parameters():
                param.grad = None

            inputs = batch["input_ids"].squeeze(dim=1).long()
            labels = batch["labels"].squeeze(dim=1).long()
            decoder_input_ids=batch["train_decoder_input_ids"].squeeze(dim=1).long()
            attention_mask = batch["attention_mask"].squeeze(dim=1).long()
            inputs, labels, attention_mask, decoder_input_ids = inputs.to(device), labels.to(device), attention_mask.to(device), decoder_input_ids.to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels, decoder_input_ids=decoder_input_ids)

            predictions = torch.argmax(outputs["logits"], dim=-1)
            if debug_mode:
                print("="*100)
                print("Inputs")
                print(bart_tokenizer.convert_ids_to_tokens(inputs[0], skip_special_tokens=False))
                print("Decoder Inputs")
                print(bart_tokenizer.convert_ids_to_tokens(decoder_input_ids[0], skip_special_tokens=False))
                print("Labels")
                print(bart_tokenizer.convert_ids_to_tokens(labels[0], skip_special_tokens=False))
                print("Training predictions")
                print(bart_tokenizer.convert_ids_to_tokens(predictions[0], skip_special_tokens=False))

            # compute accuracy over all tokens
            # (B,T), dtype bool
            train_accuracy = (predictions == labels).float()
            train_accuracy = torch.mean(attention_mask.float()*train_accuracy)/torch.mean(attention_mask.float())
            train_accuracy = train_accuracy.detach().cpu().numpy() * 100

            # compute accuracy on predicted value only
            for pred_idx, prediction in enumerate(predictions):
                ground_truth_value = batch["labels_str"][pred_idx].split("</s>")[1]
                if ground_truth_value == "":
                    ground_truth_value = "empty"
                generated_output_decoded = bart_tokenizer.decode(prediction, skip_special_tokens=False)
                try:
                    predicted_value = generated_output_decoded.split("</s>")[1]
                    if predicted_value == "":
                        predicted_value = "empty"
                except IndexError:
                    predicted_value = generated_output_decoded
                epoch_train_accuracy_only_value.append(ground_truth_value == predicted_value)
            train_accuracy_only_value = np.mean(epoch_train_accuracy_only_value)

            # compute the loss: (ignoring padding tokens)
            calculated_loss = loss_fn(outputs["logits"].view(-1, outputs["logits"].size()[-1]), labels.view(-1))
            calculated_loss.backward()
            batch_loss += calculated_loss.item()
            total_loss += calculated_loss.item()
            # gradient clipping:
            for param in model.parameters():
                param.grad.data.clamp(min=-clip_value, max=clip_value)

            optimizer.step()
            lr_scheduler.step()

            t2 = time.time()
            data_load_time = t1 - t0
            batch_time = t2 - t1

            # Print the loss values and time elapsed for every 20 batches
            if (step % 200 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Print training results
                print(f"{epoch + 1:^7} | {step:^7} || {batch_loss / batch_counts:^12.6f} | {train_accuracy: ^17.2f} | {train_accuracy_only_value: ^16.2f} || {'-':^14} || {batch_time:^9.2f} ")
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*130)
        # =======================================
        #               Evaluation
        # =======================================
        if epoch % 5 == 0:
            val_results = bart_evaluate(model=model, device=device, tokenizer=tokenizer, dataset=val_dataset, verbose=True)
        else:
            val_results = bart_evaluate(model=model, device=device, tokenizer=tokenizer, dataset=val_dataset, verbose=False)
        if debug_mode:
            train_results = bart_evaluate(model=model, device=device, tokenizer=tokenizer, dataset=train_dataset, verbose=False, debug_mode=True)
            print("Evaluation results on train data")
            print(train_results)

        val_accuracy_values = val_results["accuracy_only_values"]

        if val_accuracy_values > best_val_accuracy:
            best_val_accuracy = val_accuracy_values

            #save checkpoint
            checkpoint_name =  str(checkpoint_folder) + "/BARTv_ep_" + str(epoch) + "_valAcc_" + str(round(val_accuracy_values,2))            
            model.save_pretrained(checkpoint_name)
            checkpoint_folder_names.append(checkpoint_name)
            #print("Saved checkpoint! ", checkpoint_folder_names)

            # delete the oldest checkpoint 
            if len(checkpoint_folder_names) > 3:
                shutil.rmtree(checkpoint_folder_names[0])
                checkpoint_folder_names.pop(0)

        print(f"{epoch + 1:^7} | {'-':^7} || {avg_train_loss:^12.6f} | {train_accuracy:^17.2f} | {train_accuracy_only_value:^16.2f} || {val_accuracy_values:^14.2f}  ||  {batch_time:^9.2f}")
        print("-"*130)
        print("\n")

    print("training complete!")
    
    return checkpoint_folder_names


    
def run_training(config, device, checkpoint_folder, debug_mode:bool):
    # parameters:
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    ## Load model and tokenizer    
    if not "continue_training_checkpoint" in config.keys():
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', return_dict=True)
    else:
        model = BartForConditionalGeneration.from_pretrained(config["continue_training_checkpoint"], return_dict=True)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model.to(device)
    print("###################")
    with open(f"./datasets/{config['dataset']}/bart_seq2seq_data_train_files.json", "r", encoding="utf8") as file:
        train_files = json.load(file)
    with open(f"./datasets/{config['dataset']}/bart_seq2seq_data_val_files.json", "r", encoding="utf8") as file:
        val_files = json.load(file)

    if debug_mode:
        train_files = train_files[:4]
        val_files = val_files[:2]

    train_dataset = TextToAttributeDataset(train_files, bart_tokenizer, generate_decoder_ids=False)
    val_dataset = TextToAttributeDataset(val_files, bart_tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataset = TextToAttributeDataset(train_files, bart_tokenizer, generate_decoder_ids=True)

    print("Number of batches: ", len(train_dataloader))

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    num_training_steps = epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * 0.06)
    print(f"Have {num_training_steps} training steps and will do {num_warmup_steps} warmup steps")
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    checkpoint_folder_names = train_BART_model(
                                model=model, 
                                device=device, 
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler, 
                                tokenizer=tokenizer, 
                                train_dataloader=train_dataloader,
                                val_dataset=val_dataset,
                                train_dataset=train_dataset,
                                epochs=epochs,
                                checkpoint_folder=checkpoint_folder,
                                debug_mode=debug_mode)

    return checkpoint_folder_names

def run_bart_evaluation(run_config, device, checkpoint, save_folder, debug_mode:bool=False):
    print("Running bart tests:")
    # Load test data:
    dataset = run_config["test_dataset_name"]
    with open(run_config["test_dataset_path"], "r", encoding="utf8") as file:
        test_files = json.load(file)
    
    print("Test Data:", len(test_files))

    if debug_mode:
        print("Using only part of the test datasets:")
        test_files = test_files[:5]
    statistics = Statistics(do_collect=True)

    test_dataset = TextToAttributeDataset(test_files, bart_tokenizer, use_labels=True)

    # Load best checkpoint:
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    statistics["checkpoint_evaluated"] = checkpoint

    model.to(device)
    print(f"Evaluation on {dataset} data: ")
    bart_evaluate(model, device, bart_tokenizer, test_dataset, statistics=statistics["results"][dataset], verbose=True)
    # compute Macro F1 over dataset:
    attribute_f1_scores = []
    for attribute in statistics["results"][dataset].all_keys():
        calculate_f1_scores(statistics["results"][dataset][attribute])
        attribute_f1_scores.append(statistics["results"][dataset][attribute]["f1_score"])
    statistics["results"][dataset]["macro_f1"] = np.mean(attribute_f1_scores)
    print("F1 Score: ", np.mean(attribute_f1_scores))

    print(statistics)
    with open(os.path.join(save_folder, f"final_results_testing_on_{dataset}.json"), "w") as file:
            json.dump(statistics.to_serializable(), file, indent=5)

def run_bart_seq2seq_baseline(run_config):
    device = set_device()

    # set seeds:
    if "seed" in run_config.keys():
        torch.manual_seed(run_config["seed"])
        np.random.seed(run_config["seed"])

    # parameter:
    run_name = run_config["run_name"]
    debug_mode = run_config["debug_mode"]

    if debug_mode:
        print("#"*100)
        print("# Running in debug mode")
        print("#"*100)


    checkpoint_folder = Path("./results/bart_output/" + run_name)
    checkpoint_folder.mkdir(exist_ok=True)

    if run_config["run_bart_training"]:
        print("Running bart training:")
        checkpoint_folder_names = run_training(run_config, device, checkpoint_folder, debug_mode)

        best_checkpoint = checkpoint_folder_names[-1]
        print("Best checkpoint is: ", best_checkpoint)

    if run_config["run_bart_evaluation_on_test"]:
        print("Running bart evaluation:")
        if run_config["checkpoint_to_evaluate"] == "current_best":
            checkpoint = best_checkpoint
        else:
            checkpoint = run_config["checkpoint_to_evaluate"]
            print("Evaluation checkpoint:", checkpoint)
        run_bart_evaluation(run_config=run_config,
                            device=device,
                            checkpoint=checkpoint, 
                            save_folder=checkpoint_folder, 
                            debug_mode=debug_mode)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--config', "-c", help="Path to config file", required=True)
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, "r", encoding="utf8") as file:
        cfg = json.load(file)

    run_bart_seq2seq_baseline(run_config=cfg)
