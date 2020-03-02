import mlrose_hiive
from mlrose_hiive.runners import RHCRunner
from mlrose_hiive.runners import SARunner
from mlrose_hiive.runners import GARunner
from mlrose_hiive.runners import MIMICRunner
from mlrose_hiive import DiscreteOpt, FourPeaks
from mlrose_hiive.generators import FlipFlopGenerator
from mlrose_hiive.generators import KnapsackGenerator
from mlrose_hiive import TSPOpt


import numpy as np
import itertools as it
from collections import defaultdict


class TSPGenerator:
    @staticmethod
    def generate(seed, number_of_cities, area_width=250, area_height=250):
        np.random.seed(seed)
        x_coords = np.random.randint(area_width, size=number_of_cities)
        y_coords = np.random.randint(area_height, size=number_of_cities)

        coords = list(tuple(zip(x_coords, y_coords)))
        duplicates = TSPGenerator.list_duplicates_(coords)

        while len(duplicates) > 0:
            for d in duplicates:
                x_coords = np.random.randint(area_width, size=len(d))
                y_coords = np.random.randint(area_height, size=len(d))
                for i in range(len(d)):
                    coords[d[i]] = (x_coords[i], y_coords[i])
                    pass
            duplicates = TSPGenerator.list_duplicates_(coords)
        distances = TSPGenerator.get_distances(coords, False)

        return TSPOpt(coords=coords, distances=distances, maximize=True)


    @staticmethod
    def get_distances(coords, truncate=True):
        distances = [(c1, c2, np.linalg.norm(np.subtract(coords[c1], coords[c2])))
                     for c1, c2 in it.product(range(len(coords)), range(len(coords)))
                     if c1 != c2 and c2 > c1]
        if truncate:
            distances = [(c1, c2, int(d)) for c1, c2, d in distances]
        return distances

    #  https://stackoverflow.com/a/5419576/40410
    @staticmethod
    def list_duplicates_(seq):
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return list((indices[1:] for _, indices in tally.items() if len(indices) > 1))


class FourPeaksGenerator:
    @staticmethod
    def generate(seed, size=20, t_pct=0.15):
        np.random.seed(seed)
        fitness = FourPeaks(t_pct=t_pct)
        problem = DiscreteOpt(length=size, fitness_fn=fitness)
        return problem


def generate_problem(name, seed=None, input_size=20):
    problem = None

    ## Four Peaks
    if (name == '4peaks'):
        problem = FourPeaksGenerator.generate(seed=seed, size=input_size, t_pct=0.15)
    ## Flip Flop
    elif (name == 'flipflop'):
        problem = FlipFlopGenerator.generate(seed=seed, size=input_size)
    ## Knapsack
    elif (name == 'knapsack'):
        problem = KnapsackGenerator.generate(seed=seed, number_of_items_types=input_size,
                                            max_weight_per_item=25,
                                            max_value_per_item=10, 
                                            max_weight_pct=0.6,
                                            max_item_count=2,
                                            multiply_by_max_item_count=False)
    elif (name == 'tsp'):
        problem = TSPGenerator.generate(seed=seed, number_of_cities=input_size)

    return problem


def main():

    SEED = 1

    OUTPUT_DIRECTORY = './output'
    
    input_size = 25
    
    ## Four Peaks
    experiment_name = '4peaks'
    problem = generate_problem(experiment_name, seed=SEED, input_size=50)

    sa = SARunner(problem=problem,
                 experiment_name=experiment_name,
                 output_directory=OUTPUT_DIRECTORY,
                 seed=SEED,
                 iteration_list=2 ** np.arange(13),
                 decay_list=[mlrose_hiive.ExpDecay, mlrose_hiive.GeomDecay],
                 temperature_list=[1, 5, 10, 50],
                 max_attempts=100)

    df_run_stats, df_run_curves = sa.run() 


    ## Flip Flop
    experiment_name = 'flipflop'
    problem = generate_problem(experiment_name, seed=SEED, input_size=input_size)

    sa = SARunner(problem=problem,
                 experiment_name=experiment_name,
                 output_directory=OUTPUT_DIRECTORY,
                 seed=SEED,
                 iteration_list=2 ** np.arange(13),
                 decay_list=[mlrose_hiive.ExpDecay, mlrose_hiive.GeomDecay],
                 temperature_list=[1.0, 5, 10, 50],
                 max_attempts=100)

    # df_run_stats, df_run_curves = sa.run() 

    ## Knapsack
    experiment_name = 'knapsack'
    problem = generate_problem(experiment_name, seed=SEED, input_size=150)

    sa = SARunner(problem=problem,
                experiment_name=experiment_name,
                output_directory=OUTPUT_DIRECTORY,
                seed=SEED,
                iteration_list=2 ** np.arange(13),
                decay_list=[mlrose_hiive.ExpDecay, mlrose_hiive.GeomDecay],
                temperature_list=[1.0, 5.0, 10, 50],
                max_attempts=200)

    df_run_stats, df_run_curves = sa.run() 


if __name__ == '__main__':
    main()