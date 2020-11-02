
from graph_spectrogram import *
from sum_bandwidth_2 import *



def highest_row_intensities(spectrogram, N=3):
    """returns the indices of the N raw of highest intesity in spectrogramm"""
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


def sorting_permutattion(l):
    """return the list l sorted and the permutation used to sort it"""
    L = [(l[i], i) for i in range(len(l))]
    L.sort()
    sorted_l, permutation = zip(*L)
    return sorted_l, permutation

Xsize = 200
Ysize = 200
N = 90
#graph = create_gaussian_kernel_graph(N, Xsize=Xsize, Ysize=Ysize)
graph = create_sbm_graph(N)
#groups = create_groups_with_bfs(graph, nb_groups=3)
#spectrogram = spectrogram_with_groups(graph, groups, permutation=range(N), plot=True)
spectrogram = spectrogram_on_particular_case(graph, range(N))

indices, _ = highest_row_intensities(spectrogram, N=1)   # get the index of the row of highest intensity
_, perm = sorting_permutattion(spectrogram[indices[0]])  # get the permutation used to sort the row of highest intensity


spectrogram = spectrogram[:, perm]  # reorder the columns of the spectrogram according to the permutation
plot_matrix(spectrogram)  # plot the corresponding spectrogramm

weights = get_adjacency_matrix(graph)

random_bandwith = bandwidth_sum(np.random.permutation(N), weights) # bandwith with random permutation
print(f'band-with found with random permutation : {random_bandwith}')

bandwith_row_highest_intensity = bandwidth_sum(perm, weights)
print(f'band-with found with the permutation used to sort the row of highest intensity : {bandwith_row_highest_intensity}')

perm_mc_allister = smb2.mc_allister(weights)
print(f'value of bandwith sum found mc allister  : {bandwidth_sum(perm_mc_allister, weights)}')

#perm_local_search, _ = local_search(perm_mc_allister, graph)
#print(f'value of bandwith sum found local search from mc allister  : {bandwidth_sum(perm_local_search, weights)}')



