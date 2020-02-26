import mlrose_hiive
from mlrose_hiive.runners import RHCRunner
from mlrose_hiive.runners import SARunner

from mlrose_hiive.generators import FlipFlopGenerator
from mlrose_hiive.generators import TSPGenerator
from mlrose_hiive.generators import KnapsackGenerator

import numpy as np

SEED = 1

OUTPUT_DIRECTORY = './output'
experiment_name = 'tsp'
problem = TSPGenerator.generate(seed=SEED, number_of_cities=20)


sa = SARunner(problem=problem, 
              experiment_name=experiment_name,
              output_directory=OUTPUT_DIRECTORY,
              seed=SEED,
              iteration_list=2 ** np.arange(12),
              decay_list=[mlrose_hiive.ExpDecay, mlrose_hiive.GeomDecay],
              temperature_list=[1, 10, 50, 100],
              max_attempts=50,)

df_run_stats, df_run_curves = sa.run() 