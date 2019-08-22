import csv
import datetime
import math
import random
import sys
import argparse
from collections import Counter
import imageio
import pickle

import numpy as np
from fitness_utils.fitness import get_fitness, get_image_fitness
from genetic_components.genetic_operators import (crossover, gen_rnd_expr,
                                                  mutation,
                                                  tournament_selection)

experiment_time = datetime.datetime.now()
function_set = {
    'abs',
    'add',
    'and', 
    'cos', 
    'div', 
    'exp', 
    'if',
    'log',
    'max',
    'mdist',
    'min',
    'mod',
    'mult',
    'neg',
    'or',
    'pow',
    'scalarT',
    'scalarV',
    'sign',
    'sin',
    'sqrt',
    'sub',
    'tan',
    'warp',
    'xor',
}
terminal_set = set() 

for i in range(255):
#    terminal_set.add(i)
    terminal_set.add('x')
    terminal_set.add('y')

def initialize_population(population_size, fitness_func, image_size, image_to_fit):
    population = []
    for individual in range(population_size):
        depth_check = 0
        channel_trees = []
        for tree_number in range(4):
            while depth_check == 0:
                tree_size = int(random.random() * 4) + 1
                tree = gen_rnd_expr(function_set, terminal_set, tree_size, 'ramped half-and_half')
                depth_check = tree.get_depth()
            channel_trees.append(tree)
            individual_result = []
        red_tree = channel_trees[0]
        green_tree = channel_trees[1]
        blue_tree = channel_trees[2]
        alpha_tree = channel_trees[3]
        population.append({'channel_trees': channel_trees, 'fitness': fitness_func(x_size=image_size[0], y_size=image_size[1], red_tree=red_tree,green_tree=green_tree, blue_tree=blue_tree, alpha_tree=alpha_tree, current_individual=individual, current_generation=-1, image_to_fit=image_to_fit)})
    return population


def engine(population_size, generation_number, tournament_size, mutation_rate, crossover_rate, image_size, seed, image_to_fit=None, resume_file=None):
    engine_state = {
        'population_size': population_size, 
        'generation_number': generation_number,
        'tournament_size': tournament_size,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'image_size': image_size,
        'seed': seed,
        'image_to_fit': image_to_fit,
        'population': [],
        'current_generation': 1,
    }
    fitness_func = None
    lines = []
    lines.append(['seed', 'gen_number', 'best_fitness', 'best_individual', 'biggest_tree_depth', 'best_red', 'best_green', 'best_blue', 'best_alpha'])
    current_generation = 1
    if image_to_fit is None:
        fitness_func = get_fitness
    else:
        fitness_func = get_image_fitness
    if resume_file == None:
        population = initialize_population(population_size, fitness_func, image_size, image_to_fit)
    else:
        with open(resume_file, 'rb') as dump_file:
            engine_state = pickle.load(dump_file)
            current_generation = engine_state['current_generation']
            population = engine_state['population']
    print("Finished Generating")
    best = {'fitness': float('inf')}
    try:
        while current_generation <  generation_number:
            engine_state['population'] = population
            engine_state['current_generation'] = current_generation
            new_population = []
            new_population.append(best)
            max_tree_depth = 0
            if current_generation % 100 == 0:
                immigrants = initialize_population(population_size, fitness_func, image_size, image_to_fit)
                population.extend(immigrants)
                foo = random.sample(population, population_size)
                population = foo
            for current_individual in range(population_size - 1):
                individual_result = []
                child = [0,0,0,0] 
                max_child_depth = 0
                for current_tree in range(4):
                    member_depth = float('inf')
                    while member_depth > 17:
                        if random.random() < crossover_rate:
                            parent_1 = tournament_selection(tournament_size, population)
                            parent_2 = tournament_selection(tournament_size, population)
                            child[current_tree] = crossover(parent_1['channel_trees'][current_tree], parent_2['channel_trees'][current_tree])
                        elif random.random() < crossover_rate + mutation_rate:
                            parent = tournament_selection(tournament_size, population)
                            child[current_tree] = mutation(parent['channel_trees'][current_tree], function_set=function_set, terminal_set=terminal_set)
                        else:
                            parent = tournament_selection(tournament_size, population)
                            child[current_tree] = parent['channel_trees'][current_tree]
                        member_depth = child[current_tree].get_depth() 
                    tree_string = child[current_tree].get_string()
                    if member_depth > max_child_depth:
                        max_child_depth = member_depth 
                new_member = {}
                new_member = {'channel_trees': child, 'fitness': fitness_func(red_tree=child[0],green_tree=child[1], blue_tree=child[2], alpha_tree=child[3], current_individual=current_individual, current_generation=current_generation, x_size= image_size[0], y_size= image_size[1], best_fit=best['fitness'], image_to_fit=image_to_fit), 'depth': max_child_depth}
                if new_member['fitness'] < best['fitness']:
                    best = new_member
                    best['result'] = individual_result
                if max_tree_depth < max_child_depth:
                    max_tree_depth = max_child_depth
                new_population.append(new_member)
            lines.append([str(seed), str(current_generation), str(best['fitness']), best['depth'], max_tree_depth, best['channel_trees'][0].get_string(),best['channel_trees'][1].get_string(),best['channel_trees'][2].get_string(),best['channel_trees'][3].get_string()])
            print("###SEED " + str(seed) + " GENERATION " + str(current_generation) + " REPORT###")
            print("BEST DEPTH: " + str(best['depth']))
            print("BEST FITNESS: " + str(best['fitness']))
            print("MAX DEPTH: " + str(max_tree_depth))
            print("BEST STRINGS: \n\t" + best['channel_trees'][0].get_string() + '\n\t' + best['channel_trees'][1].get_string() + '\n\t' + best['channel_trees'][2].get_string() + '\n\t' + best['channel_trees'][3].get_string())
            population = new_population
            with open('logs/' + str(experiment_time) + '_fitness_results.csv', 'a') as writeFile:
                writer = cv.writer(writeFile)
                writer.writerows(lines)
            lines = []
            current_generation += 1
    except:
        #with open('dumps/' + str(experiment_time) + '_dumps', 'ab') as dump_file:
        with open('dumps/latest_dump', 'ab') as dump_file:
            pickle.dump(engine_state, dump_file)
            print("Saved state!")
    return True
    
def main(image_path):
    image = imageio.imread(image_path)
    image_array = np.asarray(image)
    engine(100, math.inf, 3, 0.2, 0.9, [256,256], 0, image_array)

if __name__ == "__main__":
    """ Main function worker """
    parser = argparse.ArgumentParser(
        description="Evolutionary Algorithm for Image Generation")
    parser.add_argument(
        dest="population_size",
        )
    parser.add_argument(
        dest="generation_number",
        )
    parser.add_argument(
        dest="tournament_size")
    parser.add_argument(
        dest="mutation_rate")
    parser.add_argument(
        dest="crossover_rate")
    parser.add_argument(
        help="Example of the expected format 256x256",
        dest="image_size")
    parser.add_argument(
        dest="seed")

    args = parser.parse_args()

    random.seed(int(args.seed))
    image_resolution = args.image_size.split('x')
    engine(
        int(args.population_size),
        int(args.generation_number),
        int(args.tournament_size),
        float(args.mutation_rate),
        float(args.crossover_rate),
        [int(image_resolution[0]), int(image_resolution[1])],
        int(args.seed)
    )
