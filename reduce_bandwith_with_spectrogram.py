
from sum_bandwidth_2 import *
from graph_spectrogram import plot_graph


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

def sort_highest_intensity_raw(spectrogram):
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
        print(weights)
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


def test_rearrangemant_algorithm(algorithm, nb_of_nodes=90, nb_of_test=10, plot_spectro=True, **algo_args):
    """returns the average bandwith improvement compared to mc_allister and a random permutation, by generating nb_of_tests
    graphs
    algorithm is a function which takes a spectrogram as input (and possibly other arguments which are passed in algo_args)
    and returns a permutation
    if plot spectro it will plot the spectrogramm of the first generated graph after reordering"""
    avg_improvement_random = 0
    avg_improvement_mc_allister = 0
    for i in range(nb_of_test):
        #graph = create_gaussian_kernel_graph(nb_of_nodes, Xsize=200, Ysize=200) #generate a stochastic block model graph with 3 groups
        graph = create_sbm_graph(nb_of_nodes)
        weights = get_adjacency_matrix(graph)
        #groups = create_groups_with_bfs(graph, nb_groups=3)
        groups = basic_groups(graph)
        #plot_graph(graph, groups)
        spectrogram = spectrogram_with_groups(graph, groups, permutation=range(nb_of_nodes), plot=False)
        permutation = algorithm(spectrogram, **algo_args)
        bdwth = bandwidth_sum(permutation, weights)
        bdwth_random = bandwidth_sum(np.random.permutation(nb_of_nodes), weights)
        bdwth_allister = bandwidth_sum(smb2.mc_allister(weights), weights)
        avg_improvement_random += (bdwth_random-bdwth)
        avg_improvement_mc_allister += (bdwth_allister-bdwth)
        if i == 0 and plot_spectro:
            spectrogram_example = spectrogram[:, permutation]
            plot_matrix(spectrogram_example)
    avg_improvement_mc_allister/=nb_of_test
    avg_improvement_random/=nb_of_test
    return avg_improvement_random, avg_improvement_mc_allister


rand_imp, allister_imp = test_rearrangemant_algorithm(sort_highest_intensity_raw)
print('results with algorithm which sorts the raw of highest intensity')
print('average bandwidth improvement compared to random permutation : ', rand_imp)
print('average bandwidth improvement compared to allister : ', allister_imp)

rand_imp, allister_imp = test_rearrangemant_algorithm(sorting_sum_of_rows, N=3, fix_weights=False)
print('results with algorithm which sorts the sum of 3 raws of highest intensity')
print('average bandwidth improvement compared to random permutation : ', rand_imp)
print('average bandwidth improvement compared to allister : ', allister_imp)

rand_imp, allister_imp = test_rearrangemant_algorithm(first_halves_rows, N=3)
print('results when using first halves of rows')
print('average bandwidth improvement compared to random permutation : ', rand_imp)
print('average bandwidth improvement compared to allister : ', allister_imp)



# perm_local_search, _ = local_search(perm_mc_allister, graph)
# print(f'value of bandwith sum found local search from mc allister  : {bandwidth_sum(perm_local_search, weights)}')



