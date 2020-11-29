from sum_bandwidth_2 import *
from graph_spectrogram import *


def highest_row_intensities(spectrogram, N=3):
    """returns the indices of the N rows of highest intesity in spectrogramm"""
    indices = []
    values = []
    for i in range(len(spectrogram)):
        intensity = np.sum(spectrogram[i])
        if len(values) < N or intensity > min(values):
            indices.append(i)
            values.append(intensity)
            if len(values) > N:
                for j, v in enumerate(values):
                    if v == min(values):
                        del values[j]
                        del indices[j]
                        break
    return indices, values

def sort_highest_intensity_row(spectrogram):
    indices, _ = highest_row_intensities(spectrogram, N=1)  # get the index of the row of highest intensity
    _, perm = sorting_permutation(spectrogram[indices[0]])  # get the permutation used to sort the row of highest intensity
    return perm

def sorting_permutation(l):
    """return the list l sorted and the permutation used to sort it"""
    # print(l)
    L = [(l[i], i) for i in range(len(l))]
    L.sort()
    sorted_l, permutation = zip(*L)
    return sorted_l, permutation


def sorting_sum_of_rows(spectrogram, N=3, fix_weights=False):
    """return the permutation corresponding to the order of the sum of the N most intense rows
    if fix_weights is true then the ratio beteen the importance of two following rows is forced to be 2"""
    indices, values = highest_row_intensities(spectrogram, N)
    weights = [1 for i in range(len(values))]
    if fix_weights:
        values_dict = {indices[i]: values[i] for i in range(len(indices))}
        indices.sort(key=lambda i: values_dict[i], reverse=True)
        for i in range(1, len(indices)):
            weights[i] = sum(spectrogram[indices[i - 1]]) * weights[i - 1] / (2 * sum(spectrogram[indices[i]]))

    sum_of_rows = [0 for i in range(len(spectrogram[0]))]
    for ind in range(len(indices)):
        for i in range(len(sum_of_rows)):
            sum_of_rows[i] += spectrogram[indices[ind]][i] * weights[ind]
    _, perm = sorting_permutation(sum_of_rows)
    return perm


def first_halves_rows(spectrogram, N=3):
    """Successively attributes labels to the first halves of the N most intense rows"""
    indices, values = highest_row_intensities(spectrogram, N)
    values_dict = {indices[i]: values[i] for i in range(len(indices))}
    indices.sort(key=lambda i: values_dict[i], reverse=True)
    permutation = []
    for ind in indices:
        _, row_permutation = sorting_permutation(spectrogram[ind])
        for node in row_permutation[: len(row_permutation) // 2]:
            if node not in permutation:
                permutation.append(node)
    _, last_row_permutation = sorting_permutation(spectrogram[indices[-1]])
    for n in last_row_permutation:
        if n not in permutation:
            permutation.append(n)
    return permutation


def plot_contribution_from_each_label(graph, permutation, divide_by_min=True):
    """plots the contribution of each label attributed to the graph by the permutation
    to the bandwidth sum
    if divide_by_min is True then the contributions will be divided by the minimum
    possible contribution of the labeled nodes according to their degrees"""
    N = graph.N
    nghbs = [get_neighbors(graph, i) for i in range(N)]
    labels = [0 for node in range(N)]
    for i in range(len(permutation)):
        labels[permutation[i]] = i
    contributions = [0 for i in range(N)]
    for i in range(len(permutation)):
        node = permutation[i]
        for nghb in nghbs[node]:
            contributions[i] += abs(i - labels[nghb])
        if divide_by_min:
            contributions[i] /= smb2.minimum_contribution(len(nghbs[node]))
    plt.plot([i for i in range(N)], contributions)
    plt.show()


def test_rearrangement_algorithm(algorithm, nb_repartitions=1, nb_of_nodes=90, nb_of_test=10, plot_spectro=False, **algo_args):
    """returns the average bandwith improvement compared to mc_allister and a random permutation, by generating nb_of_tests
    graphs
    algorithm is a function which takes a spectrogram as input (and possibly other arguments which are passed in algo_args)
    and returns a permutation
    if plot spectro it will plot the spectrogramm of the first generated graph after reordering"""
    avg_improvement_random = 0
    avg_improvement_mc_allister = 0
    avg_bdwth = 0
    for i in range(nb_of_test):
        #graph = create_gaussian_kernel_graph(nb_of_nodes, Xsize=200, Ysize=200) #generate a stochastic block model graph with 3 groups
        graph = create_sbm_graph(nb_of_nodes)
        weights = get_adjacency_matrix(graph)
        #groups = create_groups_with_bfs(graph, nb_groups=3)
        spectrogram = spectrogram_with_several_repartitions(graph, nb_repartitions, plot=(i==0 and plot_spectro))
        permutation = algorithm(spectrogram, **algo_args)
        bdwth = bandwidth_sum(permutation, weights)
        bdwth_random = bandwidth_sum(np.random.permutation(nb_of_nodes), weights)
        bdwth_allister = bandwidth_sum(smb2.mc_allister(weights), weights)
        avg_improvement_random += (bdwth_random-bdwth)
        avg_improvement_mc_allister += (bdwth_allister-bdwth)
        avg_bdwth += bdwth
        if i == 0 and plot_spectro:
            spectrogram_example = spectrogram[:, permutation]
            plot_matrix(spectrogram_example)
    avg_improvement_mc_allister/=nb_of_test
    avg_improvement_random/=nb_of_test
    avg_bdwth/=nb_of_test
    return avg_improvement_random, avg_improvement_mc_allister, avg_bdwth

def test_algorithms_on_same_graphs(algorithms, nb_repartitions=1, nb_of_nodes=90, nb_of_test=10, plot_densities=False) :
    """returns the average bandwith improvements compared to mc_allister and a random permutation, by generating nb_of_tests
    graphs
    algorithms is a list of algorithms to test
    these algorithms will be tested on the same graphs"""
    avg_improvements_random = [0 for ind in range(len(algorithms))]
    avg_improvements_mc_allister = [0 for ind in range(len(algorithms))]
    nb_times_best_performance = [0 for ind in range(len(algorithms))]
    bdwths_random = []
    bdwths_allister = []
    bddwths_algorithms = [[] for _ in range(len(algorithms))]
    for i in range(nb_of_test):
        print(f"Iteration {i+1} out of {nb_of_test}")
        #graph = create_gaussian_kernel_graph(nb_of_nodes, Xsize=200, Ysize=200) 
        graph = create_sbm_graph(nb_of_nodes) #generate a stochastic block model graph with 3 groups
        weights = get_adjacency_matrix(graph)
        #groups = create_groups_with_bfs(graph, nb_groups=3)
        spectrogram = spectrogram_with_several_repartitions(graph, nb_repartitions)
        bdwth_random = bandwidth_sum(np.random.permutation(nb_of_nodes), weights)
        bdwths_random.append(bdwth_random)
        bdwth_allister = bandwidth_sum(smb2.mc_allister(weights), weights)
        bdwths_allister.append(bdwth_allister)
        best_bdwth = 10 * bdwth_random
        for ind in range(len(algorithms)) :
            translation = False
            algorithm = algorithms[ind]
            if type(algorithm) == tuple :
                algorithm, translation = algorithm
            permutation = algorithm(spectrogram)
            if translation :
                permutation = best_translation(permutation, weights)
            bdwth = bandwidth_sum(permutation, weights)
            bddwths_algorithms[ind].append(bdwth)
            avg_improvements_random[ind] += (bdwth_random-bdwth)
            avg_improvements_mc_allister[ind] += (bdwth_allister-bdwth)
            if bdwth < best_bdwth :
                best_bdwth = bdwth
                best_algo = ind
        nb_times_best_performance[best_algo] += 1
    for ind in range(len(algorithms)) :
        avg_improvements_mc_allister[ind]/=nb_of_test
        avg_improvements_random[ind]/=nb_of_test
    if plot_densities:
        X = np.linspace(min(bdwths_allister)-1000, max(bdwths_random)+1000, 1000)
        print(len(X))
        kde_random = KDE(X, bdwths_random)
        plt.plot(X, kde_random, label='random')
        kde_allister = KDE(X, bdwths_allister)
        plt.plot(X, kde_allister, label='allister')
        for ind in range(len(algorithms)):
            kde_alg = KDE(X, bddwths_algorithms[ind])
            if type(algorithms[ind]) == tuple :
                name = algorithms[ind][0].__name__
                if algorithms[ind][1] :
                    name += " (T)"
            else :
                name = algorithms[ind].__name__
            plt.plot(X, kde_alg, label=name)
        plt.legend()
        plt.show()

    return avg_improvements_random, avg_improvements_mc_allister, nb_times_best_performance


def kde(x, data, h):
    """1d kernel density estimator at point x, bandwith h"""
    return sum([kernel((x - data[i]) / h) for i in range(len(data))]) / (len(data) * h)


def kernel(x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


def KDE(X, data, h=None):
    """kernel density estimator for each point in X"""
    if h is None:
        sigma = np.std(data)
        h = 1.06*sigma*len(data)**(-1/5)
    return [kde(x, data, h) for x in X]




def get_similarity(spectrogram, column1, column2, norm=1) :
    """returns a similarity between two columns of a given spectrogram
    this similarity is inferior to 1 and should be positive unless the columns are very different"""
    list_to_sum = []
    for ind in range(len(spectrogram)) :
        list_to_sum.append(abs(spectrogram[ind][column1] - spectrogram[ind][column2])**norm)
    return 1 - sum(list_to_sum) ** (1/norm)

def similarity_measure(spectrogram, permutation, norm=1) :
    """returns a similarity measure for the spectrogram with a given permutation which is the sum of similarities between each pair of neighboring columns"""
    return sum([get_similarity(spectrogram, permutation[ind], permutation[ind+1], norm) for ind in range(len(permutation)-1)])

def get_similarities(spectrogram, norm=1) :
    """computes a n*n matrix (n = the number of columns of the spectrograph) containing the similarities between each pair of its columns"""
    similarities = np.zeros((len(spectrogram), len(spectrogram[0])))
    for i in range(len(similarities)) :
        for j in range(i) :
            similarity = get_similarity(spectrogram, i, j, norm)
            similarities[i][j] = similarity
            similarities[j][i] = similarity
    return similarities

def get_relative_similarities(spectrogram, norm=1) :
    n = len(spectrogram)
    similarities = get_similarities(spectrogram, norm)
    sums_of_similarities = [sum(similarities[i]) for i in range(n)]
    relative_similarities = np.zeros((len(spectrogram), len(spectrogram[0])))
    for i in range(n) :
        for j in range(i) :
            similarity = similarities[i][j] * 2 * n / (sums_of_similarities[i] + sums_of_similarities[j])
            relative_similarities[i][j] = similarity
            relative_similarities[j][i] = similarity
    return relative_similarities

def greedy_permutation(spectrogram, norm=1, weights=None, n_iter_loc_search=20) :
    """greedily computes a permutation that should have a high similarity measure for the given spectrogram"""

    similarities = get_similarities(spectrogram, norm)
    added = [False for ind in range(len(spectrogram))]
    current_column = 0
    permutation = [0]
    added[0] = True
    while len(permutation) < len(spectrogram) :
        best_ind = current_column
        for ind in range(len(similarities[current_column])) :
            if not added[ind] and similarities[current_column][ind] > similarities[current_column][best_ind] :
                best_ind = ind
        current_column = best_ind
        permutation.append(best_ind)
        added[best_ind] = True
    return permutation


def greedy_and_local_search(spectrogram, norm=1, weights=None, n_iter_loc_search=100):
    permutation = greedy_permutation(spectrogram, norm=norm)
    return local_search_similarity_measure(spectrogram, permutation, weights=weights, nb_max_iter=n_iter_loc_search)


def local_search_similarity_measure(spectrogram, initial_solution=None, nb_max_iter=100, weights=None):
    n = len(spectrogram)
    if initial_solution is None:
        initial_solution = np.random.permutation(n)
    similarity = similarity_measure(spectrogram, initial_solution)
    S = get_similarities(spectrogram)
    best_value = similarity
    stop = False
    cpt = 0
    solution = cp.copy(initial_solution)
    similarities = []
    bandwiths = []
    while not stop and cpt < nb_max_iter:
        cpt += 1
        best_i = -1
        best_j = -1
        for i in range(n):
            for j in range(i+1, n):
                variation = 0
                if i > 0:
                    variation -= S[solution[i], solution[i-1]]
                    variation += S[solution[j], solution[i-1]]
                if j > 0:
                    variation -= S[solution[j], solution[j-1]]
                    variation += S[solution[i], solution[j-1]]
                if i < n-1:
                    variation -= S[solution[i], solution[i+1]]
                    variation += S[solution[j], solution[i+1]]
                if j < n-1:
                    variation -= S[solution[j], solution[j+1]]
                    variation += S[solution[i], solution[j+1]]
                if similarity + variation > best_value:
                    best_value = similarity + variation
                    best_i = i
                    best_j = j
        similarity = best_value
        stop = (i==-1)
        if not stop:
            solution[best_i], solution[best_j] = solution[best_j], solution[best_i]
        similarities.append(similarity)
        if weights is not None:
            bdwth = bandwidth_sum(solution, weights)
            bandwiths.append(bdwth)
    if weights is not None:
        plt.plot(range(len(bandwiths)), bandwiths)
        plt.xlabel('iteration')
        plt.ylabel('bandwith')
        plt.title('bandwith evolution while performing local search on the similarity indicator')
        plt.show()
        plt.plot(range(len(similarities)), similarities)
        plt.xlabel('iteration')
        plt.ylabel('similarity indicator')
        plt.title('local search on similarity indicator')


    return solution

def best_permutation(spectrogram, norm=1) :
    similarities = get_relative_similarities(spectrogram, norm)
    TSPpb = TSP.TSP(points=spectrogram, distances=-similarities)
    solution, _ = TSP_MIP_solving.solveIterativeSubtourEliminationGurobi(TSPpb)
    permutation = []
    #for i in range(len(solution)) :
        #print(solution[i])
    for j in range(len(solution[0])) :
        if solution[0][j] == 1 :
            permutation.append(j)
            break
    while len(permutation) < len(solution) :
        #print(permutation)
        i = permutation[-1]
        for j in range(len(solution[i])) :
            if solution[i][j] == 1 :
                permutation.append(j)
                break
    best_starting_point = permutation[0]
    worst_similarity = similarities[permutation[-1]][permutation[0]]
    for i in range(len(permutation)-1) :
        if similarities[permutation[i]][permutation[i+1]] < worst_similarity :
            best_starting_point = i+1
            worst_similarity = similarities[permutation[i]][permutation[i+1]]
    permutation = permutation[best_starting_point :] + permutation[: best_starting_point]
    return permutation

def best_translation(permutation, weights) :
    best_permutation = permutation
    best_bdwth = bandwidth_sum(permutation, weights)
    for i in range(1, len(permutation)) :
        translated_permutation = permutation[i :] + permutation[: i]
        translated_bdwth = bandwidth_sum(translated_permutation, weights)
        if translated_bdwth < best_bdwth :
            best_permutation = translated_permutation
            best_bdwth = translated_bdwth
    return best_permutation

#rand_imp, allister_imp, avg_bdwth = test_rearrangement_algorithm(local_search_similarity_measure,
                                                                 #plot_spectro=False, nb_repartitions=3)
#rand_imp, allister_imp, avg_bdwth = test_rearrangement_algorithm(greedy_permutation,
                                                                 #plot_spectro=False, nb_repartitions=3)


'''
rand_imp, allister_imp = test_rearrangement_algorithm(sort_highest_intensity_row, 10)
print('results with algorithm which sorts the row of highest intensity')
print('average bandwidth improvement compared to random permutation : ', rand_imp)
print('average bandwidth improvement compared to allister : ', allister_imp)

rand_imp, allister_imp = test_rearrangement_algorithm(sorting_sum_of_rows, 10, N=3, fix_weights=False)
print('results with algorithm which sorts the sum of 3 rows of highest intensity')
print('average bandwidth improvement compared to random permutation : ', rand_imp)
print('average bandwidth improvement compared to allister : ', allister_imp)

rand_imp, allister_imp = test_rearrangement_algorithm(first_halves_rows, 10, N=3)
print('results when using first halves of rows')
print('average bandwidth improvement compared to random permutation : ', rand_imp)
print('average bandwidth improvement compared to allister : ', allister_imp)
'''

if __name__ == '__main__' :
    algorithms = [sort_highest_intensity_row, sorting_sum_of_rows, (greedy_permutation, True), (best_permutation, True)]
    results = test_algorithms_on_same_graphs(algorithms, 3, 90, 40, plot_densities=True)
    print('results with algorithm which sorts the row of highest intensity')
    print('average bandwidth improvement compared to random permutation : ', results[0][0])
    print('average bandwidth improvement compared to allister : ', results[1][0])
    print('number of best performances : ', results[2][0])
    print('results with algorithm which sorts the sum of 3 rows of highest intensity')
    print('average bandwidth improvement compared to random permutation : ', results[0][1])
    print('average bandwidth improvement compared to allister : ', results[1][1])
    print('number of best performances : ', results[2][1])
    print('results when greedily maximizing the similarity measure')
    print('average bandwidth improvement compared to random permutation : ', results[0][2])
    print('average bandwidth improvement compared to allister : ', results[1][2])
    print('number of best performances : ', results[2][2])
    print('results when optimally maximizing the similarity measure')
    print('average bandwidth improvement compared to random permutation : ', results[0][3])
    print('average bandwidth improvement compared to allister : ', results[1][3])
    print('number of best performances : ', results[2][3])


