"""file that starts the experiments."""

from experiments.experiment2 import run_experiment_2
from experiments.experiment_wiki import run_experiment_wiki
from experiments.index_exp import generate_pdf, sheet_from_file, test_indicies, test_indicies_2


test_indicies(with_path=False)
print("starting experiment 2")
test_indicies_2(False)
# print("finished index tests")
# print("starting experiment 2")
#run_experiment_wiki()