import numpy as np
import copy as cp
from sum_bandwidth_2 import bandwidth_sum
from graph_spectrogram import *
import time


def swap_neighbour_iterator(permutation, graph):
    """iterator on neighboors of permutation
    Neighboorhood is defined as all the permutation that we can get by switching 2 vertices connected by
    an edge"""
    weights = get_adjacency_matrix(graph)
    n = graph.N
    for i in range(n):
        for j in range(i+1, n):
            if weights[i][j] != 0:

                neighbour = cp.copy(permutation)
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                vertices_connected_to_i = get_neighbors(graph, i)
                vertices_connected_to_j = get_neighbors(graph, j)
                bandwith_variation = sum([abs(neighbour[k]-neighbour[i]) for k in vertices_connected_to_i])\
                                    + sum([abs(neighbour[k]-neighbour[j]) for k in vertices_connected_to_j])\
                                    - sum([abs(permutation[k]-permutation[i]) for k in vertices_connected_to_i])\
                                    - sum([abs(permutation[k] - permutation[j]) for k in vertices_connected_to_j])
                yield neighbour, bandwith_variation


def local_search(initial_solution, graph, neighbour_iterator=swap_neighbour_iterator,
                 nb_max_iter=1000):

    weights = get_adjacency_matrix(graph)
    bandwith = bandwidth_sum(initial_solution, weights)
    temps_evaluation = 0
    best_value = 1000000
    best_sol = cp.copy(initial_solution)
    stop = False
    cpt = 0
    #print(cpt)
    while not stop and cpt < nb_max_iter:
        print('iteration', cpt)
        solution = cp.copy(best_sol)
        for nghb, bdwth_variation in neighbour_iterator(solution, graph):
            value = bandwith + bdwth_variation
            if value < best_value:
                best_value = value
                best_sol = cp.copy(nghb)
        bandwith = best_value
        stop = (solution == best_sol)
        cpt += 1
    print('temps eval', temps_evaluation)

    return best_sol, best_value


if __name__ == '__main__':
    Xsize = 100
    Ysize = 100
    N = 100
    graph = create_gaussian_kernel_graph(N, Xsize=Xsize, Ysize=Ysize)
    weights = get_adjacency_matrix(graph)
    # best_permutation = smb2.spectral_sequencing(weights)
    permutation_allister = smb2.mc_allister(weights)
    print(f'value of bandwith sum found with allister : {bandwidth_sum(permutation_allister, weights)}, solution found'
          f'{permutation_allister}')
    spectrogram_on_particular_case(graph, permutation_allister)

    initial_solution = list(range(N))
    permutation_local_search, best_val = local_search(initial_solution, graph)
    print('best_val', best_val)
    print(f'with local search {bandwidth_sum(permutation_local_search, weights)}, '
          f'solution found {permutation_local_search}')

    spectrogram_on_particular_case(graph, permutation_local_search)



