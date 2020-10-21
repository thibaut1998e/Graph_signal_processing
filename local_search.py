import numpy as np
import copy as cp
from sum_bandwidth_2 import bandwidth_sum
from graph_spectrogram import *
import time

time_to_compute_variation = 0


def swap_neighbour_iterator(permutation, graph, nghbs, swap_only_connected_vertices=False):
    """iterator on neighboors of permutation
    Neighboorhood is defined as all the permutation that we can get by switching 2 vertices connected by
    an edge"""
    n = graph.N
    for i in range(n):
        for j in range(i+1, n):
            if graph.W[i,j] != 0 or not swap_only_connected_vertices:
                neighbour = cp.copy(permutation)
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                bandwith_variation = compute_bandwith_variation(neighbour, permutation, i, j, nghbs)
                yield neighbour, bandwith_variation


def compute_bandwith_variation(res_permutation, init_permutation, i, j, nghbs):
    global time_to_compute_variation

    #vertices_connected_to_i = get_neighbors(graph, i)
    #vertices_connected_to_j = get_neighbors(graph, j)
    vertices_connected_to_i = nghbs[i]
    vertices_connected_to_j = nghbs[j]
    t = time.time()
    bandwith_variation = sum([abs(res_permutation[k] - res_permutation[i]) for k in vertices_connected_to_i]) \
                         + sum([abs(res_permutation[k] - res_permutation[j]) for k in vertices_connected_to_j]) \
                         - sum([abs(init_permutation[k] - init_permutation[i]) for k in vertices_connected_to_i]) \
                         - sum([abs(init_permutation[k] - init_permutation[j]) for k in vertices_connected_to_j])
    time_to_compute_variation += (time.time()-t)
    return bandwith_variation


def local_search(initial_solution, graph, neighbour_iterator=swap_neighbour_iterator,
                 nb_max_iter=1000):

    nghbs = [get_neighbors(graph, i) for i in range(graph.N)]
    weights = get_adjacency_matrix(graph)
    bandwith = bandwidth_sum(initial_solution, weights)
    best_value = bandwith
    best_sol = cp.copy(initial_solution)
    stop = False
    cpt = 0
    while not stop and cpt < nb_max_iter:
        print('iteration', cpt)
        solution = cp.copy(best_sol)
        for nghb, bdwth_variation in neighbour_iterator(solution, graph, nghbs):
            value = bandwith + bdwth_variation
            if value < best_value:
                best_value = value
                best_sol = cp.copy(nghb)
        bandwith = best_value
        stop = (solution == best_sol)
        cpt += 1
    return best_sol, best_value


if __name__ == '__main__':
    Xsize = 300
    Ysize = 300
    N = 100
    graph = create_gaussian_kernel_graph(N, Xsize=Xsize, Ysize=Ysize)
    weights = get_adjacency_matrix(graph)
    # best_permutation = smb2.spectral_sequencing(weights)
    permutation_allister = smb2.mc_allister(weights)
    print(f'value of bandwith sum found with allister : {bandwidth_sum(permutation_allister, weights)}, solution found'
          f'{permutation_allister}')
    spectrogram_on_particular_case(graph, permutation_allister)

    initial_solution = list(range(N))
    t = time.time()
    permutation_local_search, best_val = local_search(initial_solution, graph)
    full_time = time.time() - t

    print('best_val', best_val)
    print(f'with local search {bandwidth_sum(permutation_local_search, weights)}, '
          f'solution found {permutation_local_search}')
    print('time to compute variation', time_to_compute_variation)
    print('full time', full_time)

    spectrogram_on_particular_case(graph, permutation_local_search)




