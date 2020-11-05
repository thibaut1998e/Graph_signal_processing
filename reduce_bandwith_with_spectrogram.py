from graph_spectrogram import *
from sum_bandwidth_2 import *


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
                for j,v in enumerate(values):
                    if v == min(values):
                        del values[j]
                        del indices[j]
                        break
    return indices, values


def sorting_permutation(l):
    """return the list l sorted and the permutation used to sort it"""
    #print(l)
    L = [(l[i], i) for i in range(len(l))]
    L.sort()
    sorted_l, permutation = zip(*L)
    return sorted_l, permutation

def sorting_sum_of_rows(spectrogram, N=3, fix_weights=False) :
    """return the permutation corresponding to the order of the sum of the N most intense rows
    if fix_weights is true then the ratio beteen the importance of two following rows is forced to be 2"""
    indices, values = highest_row_intensities(spectrogram, N)
    weights = [1 for i in range(len(values))]
    if fix_weights :
        values_dict = {indices[i]: values[i] for i in range(len(indices))}
        indices.sort(key=lambda i: values_dict[i], reverse=True)
        for i in range(1, len(indices)) :
            weights[i] = sum(spectrogram[indices[i-1]]) * weights[i-1] / (2 * sum(spectrogram[indices[i]]))
        print(weights)
    sum_of_rows = [0 for i in range(len(spectrogram[0]))]
    for ind in range(len(indices)) :
        for i in range(len(sum_of_rows)) :
            sum_of_rows[i] += spectrogram[indices[ind]][i] * weights[ind]
    #print(sum_of_rows)
    return sorting_permutation(sum_of_rows)

def first_halves_rows(spectrogram, N=3) :
    """Successively attributes labels to the first halves of the N most intense rows"""
    indices, values = highest_row_intensities(spectrogram, N)
    values_dict = {indices[i]: values[i] for i in range(len(indices))}
    indices.sort(key=lambda i: values_dict[i], reverse=True)
    permutation = []
    for ind in indices :
        _, row_permutation = sorting_permutation(spectrogram[ind])
        for node in row_permutation[: len(row_permutation) // 2] :
            if node not in permutation :
                permutation.append(node)
    _, last_row_permutation = sorting_permutation(spectrogram[indices[-1]])
    for n in last_row_permutation :
        if n not in permutation :
            permutation.append(n)
    return permutation

def plot_contribution_from_each_label(graph, permutation, divide_by_min=True) :
    """plots the contribution of each label attributed to the graph by the permutation
    to the bandwidth sum
    if divide_by_min is True then the contributions will be divided by the minimum 
    possible contribution of the labeled nodes according to their degrees"""
    N = graph.N
    nghbs = [get_neighbors(graph, i) for i in range(N)]
    labels = [0 for node in range(N)]
    for i in range(len(permutation)) :
        labels[permutation[i]] = i
    contributions = [0 for i in range(N)]
    for i in range(len(permutation)) :
        node = permutation[i]
        for nghb in nghbs[node] :
            contributions[i] += abs(i - labels[nghb])
        if divide_by_min :
            contributions[i] /= smb2.minimum_contribution(len(nghbs[node]))
    plt.plot([i for i in range(N)], contributions)
    plt.show()
    
Xsize = 200
Ysize = 200
N = 90
#graph = create_gaussian_kernel_graph(N, Xsize=Xsize, Ysize=Ysize)
graph = create_sbm_graph(N)
groups = create_groups_with_bfs(graph, nb_groups=3)
spectrogram = spectrogram_with_groups(graph, groups, permutation=range(N))
spectrogram = spectrogram_on_particular_case(graph, range(N))

indices, _ = highest_row_intensities(spectrogram, N=1)   # get the index of the row of highest intensity
_, perm = sorting_permutation(spectrogram[indices[0]])  # get the permutation used to sort the row of highest intensity

_, perm2 = sorting_sum_of_rows(spectrogram, N=3, fix_weights=False)
#print(perm2)

perm3 = first_halves_rows(spectrogram, N=3)

spectrogram = spectrogram[:, perm]  # reorder the columns of the spectrogram according to the permutation
#plot_matrix(spectrogram)  # plot the corresponding spectrogramm

weights = get_adjacency_matrix(graph)

random_bandwith = bandwidth_sum(np.random.permutation(N), weights) # bandwith with random permutation
print(f'bandwidth found with random permutation : {random_bandwith}')

bandwith_row_highest_intensity = bandwidth_sum(perm, weights)
print(f'bandwidth found with the permutation used to sort the row of highest intensity : {bandwith_row_highest_intensity}')

bandwidth_sum_of_rows = bandwidth_sum(perm2, weights)
print(f'bandwidth found with the permutation used to sort the sum of the 3 highest intensity rows : {bandwidth_sum_of_rows}')

bandwidth_first_halves = bandwidth_sum(perm3, weights)
print(f'bandwidth found when using first halves of rows : {bandwidth_first_halves}')


perm_mc_allister = smb2.mc_allister(weights)
print(f'value of bandwidth sum found mc allister  : {bandwidth_sum(perm_mc_allister, weights)}')

#perm_local_search, _ = local_search(perm_mc_allister, graph)
#print(f'value of bandwith sum found local search from mc allister  : {bandwidth_sum(perm_local_search, weights)}')
