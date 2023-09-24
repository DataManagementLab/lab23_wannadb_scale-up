"""this experiment tests which vector indicies and are best suited for different amount of documents in a document base."""

import datetime
import json
import os
from wannadb.data.vector_database import compute_embedding_distances, compute_embedding_distances_withoutVDB, generate_and_store_embedding, generate_new_index

import xlsxwriter

index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}
indicies_to_test = ["FLAT",
                    "IVF_FLAT",
                    "IVF_SQ8",
                    "GPU_IVF_FLAT"]
PATH = os.getenv("DOCUMENTS_PATH")

results = {}

def test_indicies(as_json=True):
    
    generate_and_store_embedding(PATH)

    for index_type in indicies_to_test:
        index_params["index_type"] = index_type
        generate_new_index(index_params)
        try:
            result_dict = compute_embedding_distances()
            time_without, distances_without = compute_embedding_distances_withoutVDB()
            
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
    
def sheet_from_file():
    with open('index_tests.json', 'r') as fp:
        results = json.load(fp)
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
        
