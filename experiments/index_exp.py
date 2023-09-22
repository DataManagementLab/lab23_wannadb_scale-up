"""this experiment tests which vector indicies and are best suited for different amount of documents in a document base."""



import datetime
from wannadb.data.vector_database import compute_embedding_distances, compute_embedding_distances_withoutVDB, generate_and_store_embedding, generate_new_index, vectordb
from pymilvus import Collection

import xlsxwriter

index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}
indicies_to_test = ["FLAT",
                    "IVF_FLAT",
                    "IVF_SQ8"]
PATH = "C:\\Users\\Pascal\\Desktop\\WannaDB\\lab23_wannadb_scale-up\\datasets\\corona\\raw-documents"

results = {}

#generate_and_store_embedding(PATH)

for index_type in indicies_to_test:
    index_params["index_type"] = index_type
    with vectordb() as vb:
        collection = Collection('Embeddings')
        collection.release()
    generate_new_index(index_params)
    try:
        result_dict = compute_embedding_distances()
        time_without, distances_without = compute_embedding_distances_withoutVDB()
        
        results[index_type] = {"vdb":result_dict,
                            "wo_vdb":{"time":time_without,"distances":distances_without}}
    except Exception as e:
        print(e)
        print(index_type)
    
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
    
    