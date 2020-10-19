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
import random as rd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

def distance(point1, point2) :
    return np.sqrt((fst(point1) - fst(point2)) **2 + (snd(point1) - snd(point2)) **2)

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
    return weights

def get_cuthill_mckee(weights) :
    graph = csr_matrix(weights)
    order = reverse_cuthill_mckee(graph)
    permutation = [0 for i in range(len(weights))]
    for i in range(len(weights)) :
        permutation[order[i]] = i
    return permutation

def bandwidth_sum(permutation, weights) :
    bandwidth = 0
    for i in range(len(weights)) :
        for j in range(i) :
            if weights[i][j] != 0 :
                bandwidth += abs(permutation[i] - permutation[j])
    return bandwidth


def compute_edges(weights) :
    edges = []
    for i in range(len(weights)) :
        for j in range(i) :
            if weights[i][j] != 0 :
                edges.append((i, j))
    return edges

def compute_degrees(weights) :
    edges = compute_edges(weights)
    degrees = [0 for i in range(len(weights))]
    for edge in edges :
        degrees[fst(edge)] += 1
        degrees[snd(edge)] += 1
    return degrees

def compute_neighbors(weights) :
    neighbors = []
    for i in range(len(weights)) :
        neighbors_i = []
        for j in range(len(weights[i])) :
            if weights[i][j] != 0 :
                neighbors_i.append(j)
        neighbors.append(neighbors_i)
    return neighbors

def compute_lower_bound(weights) :
    n = len(weights)
    degrees = compute_degrees(weights)
    neighbors = compute_neighbors(weights)
    degrees_sorted_nodes = [i for i in range(n)]
    degrees_sorted_nodes.sort(key=lambda i: degrees[i], reverse=True)
    no_neighbors_set = []
    for node in degrees_sorted_nodes :
        no_neighbors = True
        for neighbor in neighbors[node] :
            if neighbor in no_neighbors_set :
                no_neighbors = False
                break
        if no_neighbors :
            no_neighbors_set.append(node)
    bound = sum(degrees) // 2
    print(no_neighbors_set)
    for i in no_neighbors_set :
        if degrees[i] > 2 :
            bound += degrees[i] // 2
    print("Rough lower born : {}".format(bound))
    for i in range(n) :
        if i not in no_neighbors_set :
            pseudo_degree = 0
            for j in neighbors[i] :
                if j > i and j not in no_neighbors_set :
                    pseudo_degree += 1
            if pseudo_degree > 2 :
                bound += pseudo_degree // 2
    return bound
    

import pulp as p
import gurobipy as grb
from time import time

def best_permutation_pulp(weights) :
    
    t0 = time()
    
    model = p.LpProblem("problem", p.LpMinimize)
    
    n = len(weights)
    edges = compute_edges(weights)
    print(edges)
    x_b = {(i, j): p.LpVariable("x_{}_{}".format(i, j), cat='Binary') for i in range(n) for j in range(n)}
    x = {i: p.LpVariable("x_{}".format(i), cat='Integer') for i in range(n)}
    y = {(i, j): p.LpVariable("y_{}_{}".format(i, j), cat='Integer') for (i, j) in edges}
    
    for i in range(n) :
        model += p.lpSum(x_b[(i,j)] for j in range(n)) == 1
        
    for j in range(n) :
        model += p.lpSum(x_b[(i,j)] for i in range(n)) == 1
        
    for i in range(n) :
        model += x[i] == p.lpSum(j*x_b[(i,j)] for j in range(n))
    
    for edge in edges :
        (i, j) = edge
        model += y[(i,j)] >= x[i] - x[j]
        model += y[(i,j)] >= x[j] - x[i]
        
    model += p.lpSum(y[(fst(edge),snd(edge))] for edge in edges)
    
    model.solve()
    solution = []
    for i in range(n) :
        solution.append(int(x[i].varValue))
    bandwidth = int(p.value(model.objective))
    print("La valeur optimale de la bandwidth sum est de {}.".format(bandwidth))
    print("Temps de calcul : {} secondes".format(time() - t0))
    return solution, bandwidth


# Computes the optimal solution with the gurobi linear programming solver
# Only possible up to roughly 15 nodes
def best_permutation_gurobi(weights, deltas=True, paramfile=None) :
    n = len(weights)
    n = len(weights)
    edges = compute_edges(weights)
    
    initial_solution = get_cuthill_mckee(weights)
    lower_bound = compute_lower_bound(weights)
    
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

'''
n = 14

weights = generate_graph(n, 300, 300)
default_permutation = [i for i in range(n)]
print("Bandwidth with default order : {}".format(bandwidth_sum(default_permutation, weights)))
permutation = get_cuthill_mckee(weights)
print("Heuristic permutation : {}".format(permutation))
print("Heuristic bandwidth : {}".format(bandwidth_sum(permutation, weights)))
#solution, bandwidth = best_permutation_pulp(weights)
neighbors = compute_neighbors(weights)
print(neighbors)
print({i: neighbors[i] for i in range(n)})
bound = compute_lower_bound(weights)
print("Lower bound : {}".format(bound))

solution, bandwidth = best_permutation_gurobi(weights)
solution, bandwidth = best_permutation_gurobi(weights, paramfile='gurobi_param.prm')
'''

def gurobi_tuning(weights, writefile, limit_time=600) :
    n = len(weights)
    n = len(weights)
    edges = compute_edges(weights)
    
    initial_solution = get_cuthill_mckee(weights)
    lower_bound = compute_lower_bound(weights)
    
    model = grb.Model()
    delta = {(i, j): model.addVar(vtype=grb.GRB.BINARY, name="delta_{}_{}".format(i, j)) for i in range(n) for j in range(i)}
    x = [model.addVar(vtype=grb.GRB.INTEGER, lb=0, ub=n-1, name="permutation_{}".format(i)) for i in range(n)]
    abs_diff = {edge: model.addVar(vtype=grb.GRB.INTEGER, lb=1, ub=n-1, name="abs_diff_{}".format(edge)) for edge in edges}
    
    for i in range(n) :
        x[i].setAttr(grb.GRB.Attr.Start, initial_solution[i])
        #x_b[(i, initial_solution[i])].setAttr(grb.GRB.Attr.Start, 1)

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
    
    model.setParam(grb.GRB.Param.TuneTimeLimit, limit_time)
    model.tune()
    model.getTuneResult(0)
    model.write('{}.prm'.format(writefile))

#gurobi_tuning(weights, 'gurobi_param', 1800)



