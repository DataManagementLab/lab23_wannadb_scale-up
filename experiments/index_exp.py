"""this experiment tests which vector indicies and are best suited for different amount of documents in a document base."""

import copy
import datetime
import json
import os
import time
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
from pymilvus import Collection
from wannadb.data.vector_database import EMBEDDING_COL_NAME, compute_embedding_distances, compute_embedding_distances_withoutVDB, generate_and_store_embedding, generate_new_index, vectordb

import xlsxwriter

indicies_to_test = ["FLAT",
                    "IVF_FLAT",
                    "IVF_SQ8",
                    "GPU_IVF_FLAT"]
PATH = os.getenv("DOCUMENTS_PATH")

results = {}

def test_indicies(as_json=True, with_path=True):
    
    if with_path:
        cached_document_base = generate_and_store_embedding(PATH)
    else:
        cached_document_base = generate_and_store_embedding()
    
    embedding_list = []
    attribute = cached_document_base.attributes[0]
    with vectordb() as vb:
        for embedding_name in vb._embedding_identifier:
            embedding_vector_data = attribute.signals.get(embedding_name, None)
            if embedding_vector_data is not None:
                embedding_value = embedding_vector_data.value
                
            else:
                embedding_value = np.zeros(1024)
            embedding_list.append(embedding_value)
            
    if len(embedding_list) > 0:
        combined_embedding =np.concatenate(embedding_list)

    for index_type in indicies_to_test:
        
        generate_new_index(index_type)
        try:
            document_base = copy.deepcopy(cached_document_base)
            
            with vectordb() as vdb:
                collection = Collection(EMBEDDING_COL_NAME)
                collection.load()
                start = time.time()
                _, search_time_init, update_base_time_init = vdb.compute_initial_distances(combined_embedding, document_base, True)
                duration_imit = time.time() - start
                search_time_update, update_base_time_update = vdb.updating_distances_documents(combined_embedding, document_base.documents, document_base, True)
                duration_iupdate = time.time() - start - duration_imit
                collection.release()
            
            
            results[index_type] = {"init": {"search_time":search_time_init, "update_base_time":update_base_time_init, "duration":duration_imit},
                                   "update":{"search_time":search_time_update, "update_base_time":update_base_time_update, "duration":duration_iupdate}}
        except Exception as e:
            print(e)
            print(index_type)
    
    time_without, distances_without = compute_embedding_distances_withoutVDB(cached_document_base)
    results["baseline"] = {"init": {"search_time":time_without, "update_base_time":0, "duration":time_without},
                                   "update":{"search_time":time_without, "update_base_time":0, "duration":time_without}}
    
    
    if as_json:
            
        with open('indexupdate_tests.json', 'w') as fp:
            json.dump(results, fp)
        
        generate_pdf('indexupdate_tests.json')
    
    else:  
        generate_sheet_from_json(results)
        
        
def generate_pdf(path = "indexupdate_tests.json"):
    with open(path, "r") as f:
        results = json.load(f)
        index_types = []
        index_type_durations = []
        index_typesend = []
        index_type_durationsend = []
        stype = []
        stypeend = []
        uitt = []
        for name, durations in results.items():
            for search_type, durations in durations.items():
                index_types.append(name)
                index_types.append(name)
                index_type_durations.append(durations["search_time"])
                index_type_durations.append(durations["update_base_time"])
                stype.append(search_type)
                stype.append(search_type)
                uitt.append("index")
                uitt.append("update_documents")
            
        index_types.extend(index_typesend)
        index_type_durations.extend(index_type_durationsend)
        stype.extend(stypeend)
            
        df = pd.DataFrame({"used_index":index_types, "duration":index_type_durations, "stype":stype, "used_index_time_type":uitt})
        
        labels=df['used_index'].drop_duplicates()  # set the dates as labels
        x0 = np.arange(len(labels))  # create an array of values for the ticks that can perform arithmetic with width (w)

        
        # create the data groups with a dict comprehension and groupby
        data = {''.join(k): v for k, v in df.groupby(['stype', 'used_index_time_type'])}

        # build the plots
        subs = df.stype.unique()
        stacks = len(subs)  # how many stacks in each group for a tick location
        business = df.used_index_time_type.unique()

        # set the width
        w = 0.35

        # this needs to be adjusted based on the number of stacks; each location needs to be split into the proper number of locations
        x1 = [x0 - w/stacks, x0 + w/stacks]

        fig, ax = plt.subplots()
        for x, sub in zip(x1, subs):
            bottom = 0
            for bus in business:
                height = data[f'{sub}{bus}'].duration.to_numpy()
                ax.bar(x=x, height=height, width=w, bottom=bottom)
                bottom += height
                
        ax.set_xticks(x0)
        _ = ax.set_xticklabels(labels)
        
        plt.gca().legend(('init search','init store','update search','update store'))
        
        # _, ax = plt.subplots(figsize=(7, 5))
        # sns.barplot(x="Index", y="Duration",hue="Type",data=df, ax=ax)
        ax.set_ylabel("Duration in seconds")
        ax.set_title("Durations per Index Type", size=12)
        # ax.tick_params(axis="x", labelsize=7)
        # plt.xticks(rotation=20, ha='right')
        # plt.subplots_adjust(0.09, 0.15, 0.99, 0.94)
        
        for i in ax.containers:
            ax.bar_label(i,fmt='%.2f',label_type='center')

        plt.savefig(f"Durations-per_index.pdf", format="pdf", transparent=True)
    
def sheet_from_file():
    with open('index_tests.json', 'r') as fp:
        results = json.load(fp)
        generate_sheet_from_json(results) 
        
        
        
def test_indicies_2(as_json=True):
    cached_document_base = generate_and_store_embedding(store_in_vdb=False)

    for index_type in indicies_to_test:
        
        generate_new_index(index_type)
        try:
            document_base = copy.deepcopy(cached_document_base)
            result_dict = compute_embedding_distances(document_base)
            document_base = copy.deepcopy(cached_document_base)
            time_without, distances_without = compute_embedding_distances_withoutVDB(document_base)
            
            results[index_type] = {"vdb":result_dict,
                                "wo_vdb":{"time":time_without,"distances":distances_without}}
        except Exception as e:
            print(e)
            print(index_type)
            
    if as_json:
            
        with open('index_tests.json', 'w') as fp:
            json.dump(results, fp)
    
    else:  
        generate_sheet_from_json(results)
  
def generate_sheet_from_json(results):
    timestamp = datetime.datetime.now()
    strftd = timestamp.strftime("%m_%d_%H_%M_%S")
    workbook = xlsxwriter.Workbook(f"MyExcel{strftd}.xlsx")

    for index_type, result in results.items():
        times_worksheet = workbook.add_worksheet(f"{index_type}_time")
        distances_worksheet = workbook.add_worksheet(f"{index_type}_dist")

        startrow = 1
        column = 1
        limit_arr = []

        for nprobe, probe_dict in result["vdb"].items():
            row = startrow
            
            times_worksheet.write(row, column, f"N_Probe = {nprobe}")
            distances_worksheet.write(row, column, f"N_Probe = {nprobe}")
            row +=1
            for limit, limit_dict in probe_dict.items():
                if not limit in limit_arr:
                    limit_arr.append(limit)
                
                times_worksheet.write(row, column, limit_dict["time"])
                distances_worksheet.write(row, column, limit_dict["amount_distances"])
                row +=1
            
            column += 1

        column += 1

        row = startrow
        times_worksheet.write(row, column, "Without VDB")
        distances_worksheet.write(row, column, "Without VDB")
        row += 1
        times_worksheet.write(row, column, result["wo_vdb"]["time"])
        distances_worksheet.write(row, column, result["wo_vdb"]["distances"])

        row = startrow + 1
        column = 1    
        for limit in limit_arr:
            times_worksheet.write(row, column-1, f"Limit = {limit}")
            distances_worksheet.write(row, column-1, f"Limit = {limit}")
            row +=1
            
    workbook.close()
        
