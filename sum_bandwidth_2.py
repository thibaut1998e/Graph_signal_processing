# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:56:41 2020

@author: Anthony
"""

def fst(couple) :
    a, b = couple
    return a

def snd(couple) :
    a, b = couple
    return b

import numpy as np
import matplotlib.pyplot as pl
import random as rd
import copy as cp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Calculates the distance between two points in 2 dimensions
def distance(point1, point2) :
    return np.sqrt((fst(point1) - fst(point2)) **2 + (snd(point1) - snd(point2)) **2)

# Generates a random set of points in a 2D space, then links them to obtain a weighted undirected graph
def generate_graph(numberVertices, Xsize=600, Ysize=600, theta=50, kappa=100):
    points = []
    for i in range(numberVertices) :
        newPoint = (rd.randint(0, Xsize-1), rd.randint(0, Ysize-1))
        points.append(newPoint)
    weights = []
    for i in range(numberVertices) :
        line = []
        for j in range(numberVertices) :
            if i == j :
                line.append(0)
            else :
                dist = distance(points[i], points[j])
                if dist <= kappa :
                    line.append(round(np.exp(- dist **2 / (2 * theta **2)), 2))
                else :
                    line.append(0)
        weights.append(line)
    return weights, points

# Returns the result of the Cuthill-McKee heuristic computed on a matrix of weights
def get_cuthill_mckee(weights) :
    graph = csr_matrix(weights)
    order = reverse_cuthill_mckee(graph)
    permutation = [0 for i in range(len(weights))]
    for i in range(len(weights)) :
        permutation[order[i]] = i
    return permutation

# Computes the bandwidth sum of a permutation (or linear arrangement) of nodes linked through the given weights matrix
def bandwidth_sum(permutation, weights) :
    bandwidth = 0
    for i in range(len(weights)) :
        for j in range(i) :
            if weights[i][j] != 0 :
                bandwidth += abs(permutation[i] - permutation[j])
    return bandwidth

# Returns a list of edges of a graph represented by its weights matrix
def compute_edges(weights) :
    edges = []
    for i in range(len(weights)) :
        for j in range(i) :
            if weights[i][j] != 0 :
                edges.append((i, j))
    return edges

# Returns a list of degrees for each node of a graph represented by its weights matrix
def compute_degrees(weights) :
    edges = compute_edges(weights)
    degrees = [0 for i in range(len(weights))]
    for edge in edges :
        degrees[fst(edge)] += 1
        degrees[snd(edge)] += 1
    return degrees

# Returns a list composed of a list of neighbors for each node of the graph corresponding to the weights
def compute_neighbors(weights) :
    neighbors = []
    for i in range(len(weights)) :
        neighbors_i = []
        for j in range(len(weights[i])) :
            if weights[i][j] != 0 :
                neighbors_i.append(j)
        neighbors.append(neighbors_i)
    return neighbors

# Computes a lower bound for the bandwidth sum based on the 'edges method'
def edges_lower_bound(weights) :
    n = len(weights)
    edges = compute_edges(weights)
    m = len(edges)
    bound = 0
    remaining_edges = m
    difference = 1
    while remaining_edges > n - difference :
        bound += (n - difference) * difference
        remaining_edges -= (n - difference)
        difference += 1
    bound += remaining_edges * difference
    return bound

# Computes the lowest possible value of a node to the bandwidth su according to its degree
def minimum_contribution(degree) :
    if degree % 2 == 0 :
        return (degree **2) // 4 + degree // 2
    else :
        return (degree ** 2 + 2*degree + 1) // 4

# Computes a lower bound for the bandwidth sum based on the 'vertex method'
def vertex_lower_bound(weights) :
    degrees = compute_degrees(weights)
    return sum([minimum_contribution(degree) for degree in degrees]) // 2

# Computes a lower bound for the bandwidth sum based on the 'Juvan-Mohar method'
def juvan_mohar_lower_bound(weights) :
    G = weights_to_pygsp_graph(weights)
    G.compute_fourier_basis()
    return int(G.e[1] * (len(weights) **2 - 1) / 6) + 1

# import gurobipy as grb
from time import time

# Computes the optimal solution with the gurobi linear programming solver
# Only possible up to roughly 15 nodes
def best_permutation_gurobi(weights, deltas=True, paramfile=None) :
    n = len(weights)
    n = len(weights)
    edges = compute_edges(weights)
    
    initial_solution = get_cuthill_mckee(weights)
    lower_bound = edges_lower_bound(weights)
    
    model = grb.Model()
    if deltas :
        delta = {(i, j): model.addVar(vtype=grb.GRB.BINARY, name="delta_{}_{}".format(i, j)) for i in range(n) for j in range(i)}
    else :
        x_b = {(i, j): model.addVar(vtype=grb.GRB.BINARY, name="x_{}_{}".format(i, j)) for i in range(n) for j in range(n)}
    x = [model.addVar(vtype=grb.GRB.INTEGER, lb=0, ub=n-1, name="permutation_{}".format(i)) for i in range(n)]
    abs_diff = {edge: model.addVar(vtype=grb.GRB.INTEGER, lb=1, ub=n-1, name="abs_diff_{}".format(edge)) for edge in edges}
    
    for i in range(n) :
        x[i].setAttr(grb.GRB.Attr.Start, initial_solution[i])
        #x_b[(i, initial_solution[i])].setAttr(grb.GRB.Attr.Start, 1)
    
    if not deltas :
        
        for i in range(n) :
            model.addConstr(grb.quicksum(x_b[(i, j)] for j in range(n)), grb.GRB.EQUAL, 1)
        
        for j in range(n) :
            model.addConstr(grb.quicksum(x_b[(i, j)] for i in range(n)), grb.GRB.EQUAL, 1)
        
        for i in range(n) :
            model.addConstr(x[i], grb.GRB.EQUAL, grb.quicksum(j*x_b[(i, j)] for j in range(n)))
    
    if deltas :
        
        for i in range(n) :
            for j in range(i) :
                model.addConstr(delta[(i, j)], grb.GRB.GREATER_EQUAL, (x[i] - x[j])/n)
                model.addConstr(delta[(i, j)], grb.GRB.LESS_EQUAL, 1 + (x[i] - x[j])/n)
        
        for i in range(n) :
            for j in range(i) :
                model.addConstr(x[i]-x[j]-1, grb.GRB.GREATER_EQUAL, (n+1) * (delta[(i, j)] - 1))
                model.addConstr(x[i]-x[j]+1, grb.GRB.LESS_EQUAL, (n+1) * delta[(i, j)])
    
    print("Number of edges : {}".format(len(edges)))
    for edge in edges :
        
        model.addConstr(abs_diff[edge], grb.GRB.GREATER_EQUAL, x[fst(edge)] - x[snd(edge)])
        model.addConstr(abs_diff[edge], grb.GRB.GREATER_EQUAL, x[snd(edge)] - x[fst(edge)])
        
    objective = grb.quicksum(abs_diff[edge] for edge in edges)
    #model.addConstr(objective, grb.GRB.GREATER_EQUAL, lower_bound)
    
    model.setObjective(objective)
    
    model.ModelSense = grb.GRB.MINIMIZE
    
    if paramfile != None :
        model.read(paramfile)
    
    model.optimize()
    
    solution = []
    for i in range(n) :
        solution = [int(x[i].getAttr(grb.GRB.Attr.X)) for i in range(n)]
        '''
        for j in range(n) :
            if int(x_b[(i, j)].getAttr(grb.GRB.Attr.X)) == 1 :
                solution.append(j)
        '''
    bandwidth = model.getObjective().getValue()
    
    return solution, bandwidth




import pygsp
import pygsp.graphs
import networkx

# Converts a eights matrix into a pygsp-structure graph
def weights_to_pygsp_graph(weights) :
    n = len(weights)
    weights_numpy = np.zeros((n, n))
    for i in range(n) :
        for j in range(n) :
            if weights[i][j] != 0 :
                weights_numpy[i][j] = weights[i][j]
    return pygsp.graphs.Graph(weights_numpy)

# Returns the Fiedler vector of a graph
# The Fiedler vector of a graph is the eigenvector corresponding to its 2nd smallest eigenvalue
def get_fiedler(weights) :
    edges = compute_edges(weights)
    G = networkx.Graph(edges)
    fiedler = networkx.fiedler_vector(G)
    return fiedler

# Returns the result of the spectral sequencing heuristic on a graph
def spectral_sequencing(weights) :
    n = len(weights)
    fiedler = get_fiedler(weights)
    order = [i for i in range(n)]
    order.sort(key=lambda i: fiedler[i])
    arrangement = [0 for i in range(len(weights))]
    for i in range(len(weights)) :
        arrangement[order[i]] = i
    return arrangement

# plots a 2D graph corresponding to the given weights and points
def plot_graph(weights, points) :
    n = len(weights)
    print(points)
    G = weights_to_pygsp_graph(weights)
    coordinates = np.zeros((n, 2))
    for i in range(n) :
        coordinates[i][0], coordinates[i][1] = fst(points[i]), snd(points[i])
    G.set_coordinates(coordinates)
    G.plot()

# Returns the result of the McAllister heuristic on a graph
# The starting point of the iterative process can be chosen to be random,
# or the node with the lowest degree, or a specific chose node
def mc_allister(weights, first='random') :
    t0 = time()
    n = len(weights)
    neighbors = compute_neighbors(weights)
    labels = [None for i in range(n)]
    labeled_neighbors = [[] for i in range(n)]
    unlabeled_neighbors = cp.deepcopy(neighbors)
    if first == 'random' :
        first_labeled = rd.randint(0, n-1)
    elif first == 'lowest_degree' :
        degrees = compute_degrees(weights)
        first_labeled = 0
        for i in range(n) :
            if degrees[i] < degrees[first_labeled] :
                first_labeled = i
    elif type(first) == int and first >= 0 and first < n :
        first_labeled = first
    else :
        raise Exception("Value {} is not valid for the argument 'first'".format(first))
    l = 0
    labels[first_labeled] = l
    candidate = [False for i in range(n)]
    candidate_list = []
    last_labeled = first_labeled
    for i in neighbors[first_labeled] :
        unlabeled_neighbors[i].remove(first_labeled)
        labeled_neighbors[i].append(first_labeled)
    while l < n-1 :
        l += 1
        for i in unlabeled_neighbors[last_labeled] :
            if not candidate[i] :
                candidate_list.append(i)
                candidate[i] = True
        lowest_score = 100000000
        best_candidate = candidate_list[0]
        for i in candidate_list :
            if len(unlabeled_neighbors[i]) - len(labeled_neighbors[i]) < lowest_score :
                lowest_score = len(unlabeled_neighbors[i]) - len(labeled_neighbors[i])
                best_candidate = i
        labels[best_candidate] = l
        candidate_list.remove(best_candidate)
        last_labeled = best_candidate
        for i in neighbors[best_candidate] :
            unlabeled_neighbors[i].remove(best_candidate)
            labeled_neighbors[i].append(best_candidate)
    #print("McAllister computing time : {} seconds".format(time() - t0))
    return labels

# Computes the McAllister heuristic with several starting points and
# returns the best result. One can try all the graph's nodes as starting
# points, or only the ones minimizing the degrees, or a specific number
# of strting points, with increasing degrees
def best_mc_allister(weights, research='complete') :
    n = len(weights)
    best_sum = 10000000000000
    if research == 'complete' :
        research_set = [i for i in range(n)]
    elif research == 'partial' :
        degrees = compute_degrees(weights)
        min_degree = min(degrees)
        research_set = []
        for i in range(n) :
            if degrees[i] == min_degree :
                research_set.append(i)
        print("Partial McAllister search with {} distinct starting nodes".format(len(research_set)))
    elif type(research) == int and research >= 1 and research <= n :
        degrees = compute_degrees(weights)
        research_set = [i for i in range(n)]
        research_set.sort(key=lambda i: degrees[i])
        research_set = research_set[:research]
    else :
        raise Exception("Value {} is not valid for argument 'research'".format(research))
    for i in research_set :
        labels = mc_allister(weights, i)
        labels_sum = bandwidth_sum(labels, weights)
        if labels_sum < best_sum :
            best_labels = labels
            best_sum = labels_sum
    return best_labels
'''
n = 500
weights, points = generate_graph(n, 1000, 1000)
edges = compute_edges(weights)
print("Number of nodes : {}".format(n))
print("Number of edges : {}".format(len(edges)))
edges_bound = edges_lower_bound(weights)
vertex_bound = vertex_lower_bound(weights)
juvan_mohar_bound = juvan_mohar_lower_bound(weights)
print("Edges lower bound : {}".format(edges_bound))
print("Vertex lower bound : {}".format(vertex_bound))
print("Juvan-Mohar lower bound : {}".format(juvan_mohar_bound))
default_permutation = [i for i in range(n)]
print("Bandwidth with default order : {}".format(bandwidth_sum(default_permutation, weights)))
CM_permutation = get_cuthill_mckee(weights)
#print("CM permutation : {}".format(CM_permutation))
print("CM bandwidth sum : {}".format(bandwidth_sum(CM_permutation, weights)))
SS_permutation = spectral_sequencing(weights)
#print("SS permutation : {}".format(SS_permutation))
print("SS bandwidth sum : {}".format(bandwidth_sum(SS_permutation, weights)))
mc_allister_permutation = mc_allister(weights, first='random')   
print("McAllister bandwidth with random start : {}".format(bandwidth_sum(mc_allister_permutation, weights)))
mc_allister_permutation = mc_allister(weights, first='lowest_degree')
print("McAllister bandwidth with lowest degree start : {}".format(bandwidth_sum(mc_allister_permutation, weights)))
#best_mc_allister_permutation = best_mc_allister(weights)
#print("Best McAllister (complete search) bandwidth : {}".format(bandwidth_sum(best_mc_allister_permutation, weights)))
best_mc_allister_permutation = best_mc_allister(weights, research='partial')
print("Best McAllister (partial search) bandwidth : {}".format(bandwidth_sum(best_mc_allister_permutation, weights)))
best_mc_allister_permutation = best_mc_allister(weights, research=10)
print("Best McAllister (10 starts) bandwidth : {}".format(bandwidth_sum(best_mc_allister_permutation, weights)))
'''





