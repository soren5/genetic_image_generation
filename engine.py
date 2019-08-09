import csv
import datetime
import math
import random
import sys
from collections import Counter

import numpy as np
from fitness_utils.fitness import get_fitness
from genetic_components.genetic_operators import (crossover, gen_rnd_expr,
                                                  mutation,
                                                  tournament_selection)

experiment_time = datetime.datetime.now()
function_set = {'+', '-', '*', '/', '^',}
terminal_set = set() 

for i in range(255):
    terminal_set.add(i)
    terminal_set.add('x')
    terminal_set.add('y')

def initialize_population(population_size, image_size):
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
        population.append({'channel_trees': channel_trees, 'fitness': get_fitness(x_size=image_size[0], y_size=image_size[1], red_tree=red_tree,green_tree=green_tree, blue_tree=blue_tree, alpha_tree=alpha_tree, current_individual=individual, current_generation=-1)})
    return population


def engine(population_size, generation_number, tournament_size, mutation_rate, crossover_rate, seed, image_size):
    lines = []
    lines.append(['seed', 'gen_number', 'best_fitness', 'best_individual', 'biggest_tree_depth', 'best_red', 'best_green', 'best_blue', 'best_alpha'])
    population = initialize_population(population_size, image_size)
    print("Finished Generating")
    best = {'fitness': float('inf')}
    for current_generation in range(generation_number):
        new_population = []
        new_population.append(best)
        max_tree_depth = 0
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
            new_member = {'channel_trees': child, 'fitness': get_fitness(red_tree=child[0],green_tree=child[1], blue_tree=child[2], alpha_tree=child[3], current_individual=current_individual, current_generation=current_generation, x_size= image_size[0], y_size= image_size[1]), 'depth': max_child_depth}
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
            writer = csv.writer(writeFile)
            writer.writerows(lines)
        lines = []
    return True
    
if __name__ == "__main__":
    random.seed(0)
    engine(10, 10, 3, 0.2, 0.9, 0, [255,255])
"""
if len(sys.argv) == 1:
    random.seed(0)
    engine(100, 100, 2, 0.03, 0.9, 'keras', 0)
else:
    if sys.argv[1] == 'auto':
        random.seed(0)
        engine(int(sys.argv[2]), int(sys.argv[3]), 2, 0.03, 0.9, 0)
    else:
        min_seed = input()
        max_seed = input()
        for seed in range(int(min_seed), int(max_seed)):
            random.seed(seed)
            engine(int(sys.argv[1]), int(sys.argv[2]), 2, 0.03, 0.9, seed)
#jingle()
"""
