import logging.config
import time
from experiments.util import get_documentbase, load_test_vdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

###########################################################################################################################
# Run Vector Database Loading Experiment and write Results into File
###########################################################################################################################

def run_experiment():
    '''
    Experiment: Vector Database Loading. Ensure that the preprocessed .bson file exists in the cache folder 
    or that unprocessed documents exist in the dataset folder, including COVID-19 and skyscraper as .json, and Wikipedia as .txt.
    '''

    document_base = get_documentbase('covid-19')
    start_time = time.perf_counter()
    load_test_vdb(document_base, full_embeddings=True)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    covid_results = f"COVID-19 Dataset: {execution_time}"
    print(f"COVID-19 Dataset: {execution_time}")

    '''
    document_base = get_documentbase('skyscrapers')
    start_time = time.perf_counter()
    load_test_vdb(document_base, full_embeddings=True)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    skyscraper_results = f"Skyscraper Dataset: {execution_time}"
    print(f"Skyscraper Dataset: {execution_time}")
   
    document_base = get_documentbase('wikipedia')
    start_time = time.perf_counter()
    load_test_vdb(document_base, full_embeddings=True)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    wikipedia_results = f"10k Wikipedia Dataset: {execution_time}"
    print(f"10k Wikipedia Dataset: {execution_time}")
    '''

    with open('load_vdb_experiment.txt','w') as file:
        file.write(covid_results + '\n')  
        #file.write(skyscraper_results + '\n')
        #file.write(wikipedia_results + '\n')

run_experiment()
