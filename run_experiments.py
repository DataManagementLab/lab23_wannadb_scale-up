"""file that starts the experiments."""

#from experiments.experiment2 import run_experiment_2
#from experiments.experiment_wiki import run_experiment_wiki
#from experiments.index_exp import test_indicies#, generate_pdf, sheet_from_file, test_indicies_2
from experiments.single_distance_experiments import run_single_distance


run_single_distance()
#test_indicies(with_path=False)
# print("finished index tests")
# print("starting experiment 2")
#run_experiment_wiki()