import random 
import math
from collections import Counter
import copy
from .node import Node

special_case_3_child = {'if'}
special_case_1_child = {'abs', 'cos', 'sin', 'tan', 'neg', 'exp', 'log', 'sign'}
def crossover(parent_1, parent_2):
    crossover_node = None
    if random.random() < 0.9:
        #function crossover
        parent_1_candidates = get_candidates(parent_1, True) 
        parent_1_chosen_node = random.choice(list(parent_1_candidates.elements()))
        possible_children = []
        for i in range(len(parent_1_chosen_node.children)):
            if parent_1_chosen_node.children[i].terminal == False:
                possible_children.append(i)
        if possible_children != []:
            crossover_node = copy.deepcopy(parent_1_chosen_node.children[random.sample(possible_children, 1)[0]])
        else:
            crossover_node = copy.deepcopy(parent_1_chosen_node)
    else:
        parent_1_terminals = get_terminals(parent_1) 
        crossover_node = random.choice(list(parent_1_terminals.elements()))
    if crossover_node == None:
        print("ERROR: Did not select a crossover node")
    new_individual = copy.deepcopy(parent_2)
    parent_2_candidates = get_candidates(new_individual, True)
    parent_2_chosen_node = random.choice(list(parent_2_candidates.elements()))
    parent_2_chosen_node.children[random.randint(0, len(parent_2_chosen_node.children) - 1)] = crossover_node
    return new_individual

def tournament_selection(tournament_size, population):
    winner = {'fitness': float('inf')}
    while winner['fitness'] == float('inf'):
        tournament_population = random.sample(population, tournament_size)
        for i in tournament_population:
            if i['fitness'] < winner['fitness']:
                winner = i
    return winner

def mutation(parent, function_set, terminal_set):
        new_individual = copy.deepcopy(parent)
        candidates = get_candidates(new_individual, True) 
        chosen_node = random.choice(list(candidates.elements()))
        mutation_node = gen_rnd_expr(function_set=function_set, terminal_set=terminal_set, max_depth=chosen_node.get_depth(1), method='ramped half-and-half') 
        chosen_node.children[random.randint(0, len(chosen_node.children) - 1)] = mutation_node
        return new_individual

def get_candidates(node, root):
    candidates = Counter()
    for i in node.children:
        if i != None and i.terminal == False:
            candidates.update([node])
            candidates.update(get_candidates(i, False))
    if root and candidates == Counter():
        candidates.update([node])
    return candidates

def get_terminals(node):
    candidates = Counter()
    if node.terminal:
        candidates.update([node])
    else:
        for i in node.children:
            if i != None:
                candidates.update(get_terminals(i))
    return candidates

def gen_rnd_expr(function_set, terminal_set, max_depth, method):
    if method == 'ramped half-and-half':
        children = [gen_rnd_expr(function_set, terminal_set, max_depth - 1, 'grow'), gen_rnd_expr(function_set, terminal_set, max_depth - 1, 'full')]
        value = random.sample(function_set, 1)[0]
        #Special case for operators with 3 children
        if value in special_case_3_child:
            children.append(gen_rnd_expr(function_set, terminal_set, max_depth - 1, 'grow'))
        #Special case for operators with 1 child
        if value in special_case_1_child:
            children = [children[0]]
        node = Node(
            value=value,
            terminal=False,
            children=children)
        return node 
    if max_depth == 0 or (method == 'grow' and random.random() < (len(terminal_set) / (len(terminal_set) + len(function_set)))):
        return Node(value=random.sample(terminal_set, 1)[0], terminal=True, children=[])        
    else:
        children = [gen_rnd_expr(function_set, terminal_set, max_depth - 1, method), gen_rnd_expr(function_set, terminal_set, max_depth - 1, method)]
        value = random.sample(function_set, 1)[0]
        #Special case for operators with 3 children
        if value in special_case_3_child:
            children.append(gen_rnd_expr(function_set, terminal_set, max_depth - 1, 'grow'))
        #Special case for operators with 1 child
        if value in special_case_1_child:
            children = [children[0]]

        node = Node(value=value,
        terminal=False,
        children=children)
        #Uncomment to surpress silent mutations
        """
        if left_child.set_name == 'terminal' and right_child.set_name == 'terminal' and left_child.value != 'x' and right_child.value != 'x':
            return Node(value=execute_tree({}, node), left_child= None, right_child= None,set_name= 'terminal')
        else:
        """
        return node
