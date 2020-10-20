import numpy as np
import copy as cp
from sum_bandwidth_2 import bandwidth_sum
from graph_spectrogram import *


def swap_neighbour_iterator(permutation, weights):
    """iterator on neighboors of permutation
    Neighboorhood is defined as all the permutation that we can get by switching 2 vertices connected by
    an edge"""
    n = len(weights)
    for i in range(n):
        for j in range(i+1, n):
            if weights[i][j] != 0:
                neighbour = cp.copy(permutation)
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                yield neighbour


def local_search(initial_solution, weights, func_to_optimize=bandwidth_sum, neighbour_iterator=swap_neighbour_iterator,
                 nb_max_iter=1000):

    best_value = 1000000
    best_sol = cp.copy(initial_solution)
    stop = False
    cpt = 0
    while not stop and cpt < nb_max_iter:
        print('iteration', cpt)
        solution = cp.copy(best_sol)
        for nghb in neighbour_iterator(solution, weights):
            value = func_to_optimize(nghb, weights)
            if value < best_value:
                best_value = value
                best_sol = cp.copy(nghb)

        stop = (solution == best_sol)
        cpt += 1

    return best_sol, best_value


if __name__ == '__main__':
    Xsize = 100
    Ysize = 100
    N = 50
    graph = create_gaussian_kernel_graph(N, Xsize=Xsize, Ysize=Ysize)
    weights = get_adjacency_matrix(graph)
    # best_permutation = smb2.spectral_sequencing(weights)
    permutation_allister = smb2.mc_allister(weights)
    print(f'value of bandwith sum found with allister : {bandwidth_sum(permutation_allister, weights)}, solution found'
          f'{permutation_allister}')
    spectrogram_on_particular_case(graph, permutation_allister)

    initial_solution = list(range(N))
    permutation_local_search, best_val = local_search(initial_solution, weights)
    print(f'with local search {bandwidth_sum(permutation_local_search, weights)}, '
          f'solution found {permutation_local_search}')

    spectrogram_on_particular_case(graph, permutation_local_search)



