"""this experiment tests which vector indicies and are best suited for different amount of documents in a document base."""

import copy
import datetime
import json
import os
import time

import numpy as np
from pymilvus import Collection
from wannadb.data.vector_database import EMBEDDING_COL_NAME, compute_embedding_distances, compute_embedding_distances_withoutVDB, generate_and_store_embedding, generate_new_index, vectordb

import xlsxwriter

indicies_to_test = ["FLAT",
                    "IVF_FLAT",
                    "IVF_SQ8",
                    "GPU_IVF_FLAT"]
PATH = os.getenv("DOCUMENTS_PATH")

results = {}

def test_indicies(as_json=True):
    
    cached_document_base = generate_and_store_embedding(PATH)
    
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
                vdb.compute_inital_distances(combined_embedding, document_base)
                duration_imit = time.time() - start
                vdb.updating_distances_documents(combined_embedding, document_base.documents, document_base)
                duration_iupdate = time.time() - start - duration_imit
                collection.release()
            
            
            results[index_type] = {"init":duration_imit,
                                   "update":duration_iupdate}
        except Exception as e:
            print(e)
            print(index_type)
            
    if as_json:
            
        with open('indexupdate_tests.json', 'w') as fp:
            json.dump(results, fp)
    
    else:  
        generate_sheet_from_json(results)
    
def sheet_from_file():
    with open('index_tests.json', 'r') as fp:
        results = json.load(fp)
        generate_sheet_from_json(results) 
        
        
        
def test_indicies_2(as_json=True):
    cached_document_base = generate_and_store_embedding(PATH)

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
        
