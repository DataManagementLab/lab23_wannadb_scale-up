import logging.config
import time
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from pymilvus import Collection
from wannadb.data.data import DocumentBase
from wannadb.matching.distance import SignalsMeanDistance
from wannadb.statistics import Statistics
from wannadb.data.vector_database import EMBEDDING_COL_NAME, vectordb
from wannadb.data.signals import  LabelEmbeddingSignal, CurrentMatchIndexSignal, CachedDistanceSignal, CombinedEmbeddingSignal
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile, pstats, io
from pstats import SortKey
from experiments.util import load_test_vdb, get_documentbase, compute_distances_and_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

#########################################################################################################
# Collection of single vector similarity search experiments
#########################################################################################################

'''
Contains a Collection of Experiment Scripts:
Experiment 1: Comparison of single vector similarity search with one iteration of previous distance computation (without storing distances, only one EmbeddingSignal)
Experiment 2: Comparison of single vector similarity search with one iteration of previous distance computation (with storing distances, only one EmbeddingSignal)
Experiment 3: Comparison of single vector similarity search with one iteration of previous distance computation (with storing distances, all three EmbeddingSignals)
Experiment 4: Vector database run-time comparison of similarity metrics (Inner product vs. Cosine, all three EmbeddingSignals)
Experiment 5: Run-Time comparison updating vector database vs. initial updating approach (without storing distances, all three EmbeddingSignals)
Experiment 6: Run-Time comparison single similarity search vs. range search (without storing distances, all three EmbeddingSignals)

Ensure that the preprocessed .bson file exists in the cache folder or that unprocessed documents exist in the dataset folder, 
including COVID-19 and skyscraper as .json, and Wikipedia as .txt.
'''

NUM_SEARCH_RUNS = 10 #computing average over 10 search iterations
LIMIT = 16384 #vector similarity search returns top 16384 results

def run_experiment1(document_base: DocumentBase, index_types: List[str] = ["FLAT","IVF_FLAT","IVF_SQ8"]):
    '''
    Experiment 1 - Comparison of single distance computation - only LabelEmbedding - without storing distance
    '''
    experiment_results = {}
    attribute = document_base.attributes[0]

    with vectordb() as vb:
        for i in index_types:
            collection = Collection(EMBEDDING_COL_NAME)
            
            #Adjust vector index type
            vb.regenerate_index(i)
            collection.load()

            if i in ["IVF_FLAT", "IVF_SQ8", "GPU_IVF_FLAT"]:
                vb._search_params["nprobe"] = 120
            
            #execute vector similarity search
            search_times = []
            for _ in range(NUM_SEARCH_RUNS):
                start_time = time.perf_counter()
                results = collection.search(
                    data=[attribute[LabelEmbeddingSignal]], 
                    anns_field="embedding_value", 
                    param=vb._search_params,
                    limit=LIMIT,
                    expr=None,
                    output_fields=['id', 'document_id'],
                    consistency_level="Strong"
                )
                end_time = time.perf_counter()
                search_times.append(end_time - start_time)

            avg_time = sum(search_times) / NUM_SEARCH_RUNS
            experiment_results[i] = avg_time

            collection.release()

        overall, search, store = compute_distances_and_store(document_base, attribute)
        experiment_results['without vdb'] = overall

    index_type = list(experiment_results.keys())
    run_times = list(experiment_results.values())

    # Plot results
    plt.figure(figsize=(10, 6)) 
    plt.bar(index_type, run_times, color="#0c2461")
    plt.xlabel('Index Type')
    plt.ylabel('Run-Time (in seconds)')
    plt.title('Run-times per index type - without storing, only LabelEmbeddings')
    plt.savefig('experiment1.pdf')
    plt.clf()


def run_experiment2(document_base: DocumentBase, index_types: List[str] = ["FLAT", "IVF_FLAT", "IVF_SQ8"]): 
    '''
    Experiment 2 - Comparison of single distance computation - only LabelEmbedding - with storing distance
    '''
    experiment_results = {"used_index": [], "duration": []}
    attribute = document_base.attributes[0]

    with vectordb() as vb:
        for i in index_types:
            collection = Collection(EMBEDDING_COL_NAME)
            try:
                collection.release()
            except:
                pass
            
            #Adjust vector index type
            vb.regenerate_index(i)
            collection.load()

            if i in ["IVF_FLAT", "IVF_SQ8", "GPU_IVF_FLAT"]:
                vb._search_params["nprobe"] = 120

            remaining_documents = []
            n_docs = len(document_base.documents)

            #execute vector similarity search
            search_time = []
            for _ in range(NUM_SEARCH_RUNS):
                start_time = time.perf_counter()
                results = collection.search(
                    data=[attribute[LabelEmbeddingSignal]], 
                    anns_field="embedding_value", 
                    param=vb._search_params,
                    limit=LIMIT,
                    expr=None,
                    output_fields=['id', 'document_id'],
                    consistency_level="Strong"
                )

                for j in results[0]:
                    if len(remaining_documents) < n_docs:
                        current_document = document_base.documents[j.entity.get('document_id')]
                        if current_document not in remaining_documents:
                            current_nugget = j.entity.get('id')
                            current_document[CurrentMatchIndexSignal] = CurrentMatchIndexSignal(current_nugget)
                            current_document.nuggets[current_nugget][CachedDistanceSignal] = CachedDistanceSignal(1 - j.distance)
                            remaining_documents.append(current_document)
                    else:
                        break
                end_time = time.perf_counter()
                search_time.append(end_time - start_time)

            avg_time = sum(search_time) / NUM_SEARCH_RUNS
            experiment_results['duration'].append(avg_time)
            experiment_results['used_index'].append(i)

            collection.release()

        # Compute initial distances - only LabelEmbedding - without vector database
        overall_time, search_time, storing_time = compute_distances_and_store(document_base, attribute, full_embeddings=False)
        experiment_results['duration'].append(overall_time)
        experiment_results['used_index'].append('without vdb')

    index_types = experiment_results["used_index"]
    values = experiment_results['duration']

    #Plot results
    plt.bar(index_types, values, color='#0c2461')
    plt.xlabel('Index Type')
    plt.ylabel('Run-Time (in seconds)')
    plt.title('Initial Distance Computation/Storing LabelEmbedding (AVG of 10 Iterations)')
    plt.savefig('experiment2.pdf')
    plt.clf()


def run_experiment3(document_base: DocumentBase, index_types: List[str] = ["FLAT", "IVF_FLAT", "IVF_SQ8"]):
    '''
    Experiment 3 - Comparison of single distance computation - full embeddings Label, Text, SentenceContextCache - with storing distance
    '''
    experiment_results = {"used_index": [], "duration": [], "storing": []}
    attribute = document_base.attributes[0]

    with vectordb() as vb:
        for i in index_types:
            collection = Collection(EMBEDDING_COL_NAME)
            try:
                collection.release()
            except:
                pass
            
            #Adjust vector index type
            vb.regenerate_index(i)
            collection.load()

            if i in ["IVF_FLAT", "IVF_SQ8", "GPU_IVF_FLAT"]:
                vb._search_params["nprobe"] = 120
            
            combined_embedding = np.concatenate([attribute[LabelEmbeddingSignal], np.zeros(2048)])

            #Execute vector similarity search
            runtimes = []
            storings = []
            for _ in range(NUM_SEARCH_RUNS):
                _, search_time, store_time = vb.compute_initial_distances(attribute_embedding=combined_embedding, document_base=document_base, return_times=True)
                runtimes.append(search_time)
                storings.append(store_time)

            avg_runtime = sum(runtimes) / NUM_SEARCH_RUNS
            avg_storing = sum(storings) / NUM_SEARCH_RUNS

            experiment_results['duration'].append(avg_runtime)
            experiment_results['storing'].append(avg_storing)
            experiment_results['used_index'].append(i)

            collection.release()

        # Compute initial distances - all three Embeddings - without vector database
        experiment_results['used_index'].append('without vdb')

        overall, search_time, storing_time = compute_distances_and_store(document_base, attribute, full_embeddings=True)
        experiment_results['duration'].append(search_time)
        experiment_results['storing'].append(storing_time)
        
        #Plot results
        df = pd.DataFrame(experiment_results)

        sns.set(style="whitegrid")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='used_index', y='duration', data=df, color='skyblue', ax=ax)
        sns.barplot(x='used_index', y='storing', data=df, color='orange', ax=ax)

        ax.set(xlabel='Used Index', ylabel='Run-Time (in seconds)')
        ax.legend(handles=[plt.Rectangle((0,0),1,1,fc='skyblue'), plt.Rectangle((0,0),1,1,fc='orange')],
                labels=['Index Run-Time', 'Storing Run-Time'])
        ax.set_title("Single initial distance computation (one attribute, full embedding signals)")

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

        plt.savefig('experiment3.pdf')
        plt.clf()

def run_experiment4(document_base: DocumentBase, index_types: List[str] = ["FLAT", "IVF_FLAT", "IVF_SQ8"], metrics: List[str] = ['IP', 'COSINE']):
    '''
    Experiment 4 - Comparison of different similarity metrics: IP vs. Cosine
    '''
    experiment_results = {"used_index": [], "metric_type": [], "run_time": []}
    attribute = document_base.attributes[0]

    with vectordb() as vb:
        for metric in metrics:
            for index_type in index_types:
                collection = Collection(EMBEDDING_COL_NAME)
                try:
                    collection.release()
                except:
                    pass
                
                #Adjust vector index type and similarity metric
                vb.regenerate_index(index_name=index_type, metric_type=metric)
                vb._search_params["metric_type"] = metric

                if index_type in ["IVF_FLAT", "IVF_SQ8", "GPU_IVF_FLAT"]:
                    vb._search_params["nprobe"] = 120
        
                combined_embedding = np.concatenate([attribute[LabelEmbeddingSignal], np.zeros(2048)])
                collection.load()

                #Execute vector similarity search
                runtimes = []
                for _ in range(NUM_SEARCH_RUNS):
                    _, search_time, store_time = vb.compute_initial_distances(attribute_embedding=combined_embedding, document_base=document_base, return_times=True)
                    runtimes.append(search_time)

                avg_runtime = sum(runtimes) / NUM_SEARCH_RUNS
                
                experiment_results['run_time'].append(avg_runtime)
                experiment_results['metric_type'].append(metric)
                experiment_results['used_index'].append(index_type)

                collection.release()
    
    df = pd.DataFrame(experiment_results)

    #Plot results
    sns.barplot(x="used_index", y="run_time", hue="metric_type", data=df, palette='Blues')
    plt.xlabel('Index Type')
    plt.ylabel('Run-Time (in seconds)')
    plt.title('Run-Time IP vs. COSINE (AVG 10 Iterations)')
    plt.savefig('experiment4.pdf')
    plt.clf()

def run_experiment5(document_base: DocumentBase, index_types: List[str] = ["FLAT", "IVF_FLAT", "IVF_SQ8"]):
    '''
    Experiment 5 - Comparison Updating VDB vs. Initial Updating Approach
    '''
    experiment_results = {"used_index": [], "duration": [], "storing": []}

    with vectordb() as vb:
        for i in index_types:
            collection = Collection(EMBEDDING_COL_NAME)
            try:
                collection.release()
            except:
                pass
            
            #Adjust vector index type
            vb.regenerate_index(i)
            collection.load()

            if i in ["IVF_FLAT", "IVF_SQ8", "GPU_IVF_FLAT"]:
                vb._search_params["nprobe"] = 120
            
            combined_embedding = document_base.nuggets[0][CombinedEmbeddingSignal]

            runtimes = []
            storings = []

            for _ in range(NUM_SEARCH_RUNS):
                search_time, store_time = vb.updating_distances_documents(target_embedding=combined_embedding, documents=document_base.documents, document_base=document_base, return_times=True)
                runtimes.append(search_time)
                storings.append(store_time)

            avg_runtime = sum(runtimes) / NUM_SEARCH_RUNS
            avg_storing = sum(storings) / NUM_SEARCH_RUNS

            experiment_results['duration'].append(avg_runtime)  
            experiment_results['storing'].append(avg_storing)
            experiment_results['used_index'].append(i)

            collection.release()

    # Compute initial distances - all three Embeddings - without vector database
    experiment_results['used_index'].append('without vdb')

    overall_time, search_time, storing_time = compute_distances_and_store(document_base, document_base.nuggets[0], full_embeddings=True)
    experiment_results['duration'].append(search_time)
    experiment_results['storing'].append(storing_time)

    #Plot results
    df = pd.DataFrame(experiment_results)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x='used_index', y='duration', data=df, color='skyblue', ax=ax)
    sns.barplot(x='used_index', y='storing', data=df, color='orange', ax=ax)

    ax.set(xlabel='Used Index', ylabel='Run-Time (in seconds)')
    ax.legend(handles=[plt.Rectangle((0,0),1,1,fc='skyblue'), plt.Rectangle((0,0),1,1,fc='orange')],
              labels=['Index Run-Time', 'Storing Run-Time'])
    ax.set_title("Update computation (AVG of 10 Iterations)")

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.savefig('experiment5.pdf')
    plt.clf()
    
def run_experiment6(document_base: DocumentBase, index_types: List[str] = ["FLAT", "IVF_FLAT", "IVF_SQ8"]):
    '''
    Experiment 6 - Comparison single similarity search vs. range search
    '''
    experiment_results = {"used_index": [], "run_time": [], "search_type": []}
    attribute = document_base.attributes[0]

    with vectordb() as vb:
        for i in index_types:
            collection = Collection(EMBEDDING_COL_NAME)
            try:
                collection.release()
            except:
                pass
            
            #Adjust vector index
            vb.regenerate_index(i)
            collection.load()

            if i in ["IVF_FLAT", "IVF_SQ8", "GPU_IVF_FLAT"]:
                vb._search_params["nprobe"] = 120
            
            combined_embedding = np.concatenate([attribute[LabelEmbeddingSignal], np.zeros(2048)])

            # Execute single vector similarity search
            search_times = []
            for _ in range(NUM_SEARCH_RUNS):
                start_time = time.perf_counter()
                results = collection.search(
                    data=[combined_embedding], 
                    anns_field="embedding_value", 
                    param=vb._search_params,
                    limit=LIMIT,
                    expr=None,
                    output_fields=['id','document_id'],
                    consistency_level="Strong"
                )

                end_time = time.perf_counter()
                execution_time = end_time - start_time
                search_times.append(execution_time)

            avg_times = sum(search_times) / NUM_SEARCH_RUNS
            experiment_results['run_time'].append(avg_times)  
            experiment_results['used_index'].append(i)
            experiment_results['search_type'].append('single')

            # Execute 4 range searches
            search_times = []
            for _ in range(NUM_SEARCH_RUNS):
                start_range = 0.8
                end_range = 1.0
                search_limit = LIMIT

                start_time = time.perf_counter()

                while search_limit > 0: 
                    search_param = {
                        "data": [combined_embedding], 
                        "anns_field": "embedding_value", 
                        "param": {"metric_type": "COSINE", "params": {"radius": start_range, "range_filter": end_range}, "offset": 0},
                        "limit": search_limit,
                        "output_fields": ["id", "document_id"]
                    }

                    res = collection.search(**search_param)
                    start_range -= 0.2
                    end_range -= 0.2
                        
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                search_times.append(execution_time)

            avg_times = sum(search_times) / NUM_SEARCH_RUNS
            experiment_results['run_time'].append(avg_times)  
            experiment_results['used_index'].append(i)
            experiment_results['search_type'].append('range')

            collection.release()

    #Plot results
    df = pd.DataFrame(experiment_results)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(x='used_index', y='run_time', hue='search_type', data=df, palette='Blues')
    plt.xlabel('Index Type')
    plt.ylabel('Run-Time (in seconds)')
    plt.title('Single Search vs. Range Search (AVG 10 Iterations)')
    plt.savefig('experiment6.pdf')
    plt.clf()
    
def run_single_distance():

    document_base= get_documentbase('covid-19')
    # load_test_vdb(document_base)
    # run_experiment1(document_base=document_base)
    # run_experiment2(document_base=document_base)
    load_test_vdb(document_base=document_base, full_embeddings=True)
    # run_experiment3(document_base=document_base)
    # run_experiment4(document_base=document_base)
    # run_experiment5(document_base=document_base)
    run_experiment6(document_base=document_base)







        


            
            
                
            
                
                

                
            



        

