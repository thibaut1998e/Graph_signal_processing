import numpy as np
import pygsp
import os
import matplotlib.pyplot as plt
#import sum_bandwidth as smb
import sum_bandwidth_2 as smb2


np.random.seed(0)
OUTPUT_DIR = "output/"

"""
    Creates the directory for the given file name if it does not exist.
    --
    In:
        * file_name: Directory to create, or file for which we want to create a directory.
    Out:
        * None.
"""


def create_directory_for(file_name):
    # Creates the corresponding directory
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)


"""
    Creates a path graph.
    --
    In:
        * graph_order: Number of vertices.
    Out:
        * graph: A PyGSP path graph.
"""


def create_path_graph(graph_order):
    # PyGSP function
    graph = pygsp.graphs.Path(graph_order)
    graph.compute_fourier_basis()
    return graph


"""
    Creates a stochastic block model.
    --
    In:
        * graph_order: Number of vertices.
    Out:
        * graph: A PyGSP path graph.
"""


def create_sbm_graph(graph_order):
    # PyGSP function
    groups = np.array(
        [0] * (graph_order // 3) + [1] * (graph_order // 3) + [2] * (graph_order - 2 * graph_order // 3))
    graph = pygsp.graphs.StochasticBlockModel(graph_order, k=3, z=groups, p=[0.4, 0.6, 0.3], q=0.02)
    graph.set_coordinates(kind="spring")
    graph.compute_fourier_basis()
    return graph


"""
    Creates a sensor graph.
    --
    In:
        * graph_order: Number of vertices.
    Out:
        * graph: A PyGSP sensor graph.
"""


def create_sensor_graph(graph_order):
    # PyGSP function
    graph = pygsp.graphs.Sensor(graph_order, seed=np.random.randint(2 ** 8))
    graph.compute_fourier_basis()
    return graph


def create_gaussian_kernel_graph(nb_vertices, **kwargs):
    """generate a graph with weights defined by a gaussian kernel
    kwargs contains X_size, Y_size, theta, kappa"""
    weights = smb.generate_graph(nb_vertices, **kwargs)
    graph = create_graph_with_weights(weights)
    return graph


def create_graph_with_weights(weights):
    """generate a graph with a given adjacency matrix (weights)"""
    weights = np.array(weights)
    graph = pygsp.graphs.Graph(weights)
    graph.set_coordinates()
    graph.compute_fourier_basis()
    return graph


"""
    Plots a PyGSP graph.
    --
    In:
        * graph: Graph to plot.
        * signal: Signal to plot on vertices.
        * title: Figure title.
        * file_name: Where to save the results.
    Out:
        * None.
"""


def plot_graph(graph, signal=None, title="", file_name=None):
    # With or without signal
    figure = plt.figure(figsize=(20, 10))
    if signal is None:
        graph.plot(ax=figure.gca())
    else:
        graph.plot_signal(signal, ax=figure.gca())

    # Plot
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Save
    if file_name is not None:
        figure.savefig(file_name, bbox_inches="tight")


"""
    Plots a matrix.
    --
    In:
        * matrix: Matrix to plot.
        * rows_labels: Labels associated with the rows.
        * cols_labels: Labels associated with the columns.
        * title: Figure title.
        * colorbar: Set to True to plot a colorbar.
        * round_values: Set to >= 0 to plot values in matrix cells.
        * file_name: Where to save the results.
    Out:
        * None.
"""


def create_heat_kernel(graph, scale):
    # PyGSP kernel
    kernel = pygsp.filters.Heat(graph, scale, normalize=True)
    return kernel


def plot_matrix(matrix, rows_labels="", cols_labels="", rows_title="", cols_title="", title="", colorbar=False,
                round_values=None, file_name=None):
    # Plot matrix
    figure, axis = plt.subplots(figsize=(20, 20))
    cax = axis.matshow(matrix)

    # Add colorbar
    if colorbar:
        plt.colorbar(cax)

    # Add values
    if round_values is not None:
        color_change_threshold = 0.5 * (np.max(matrix) + np.min(matrix))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = round(matrix[i, j], round_values) if round_values > 0 else int(matrix[i, j])
                color = "black" if matrix[i, j] > color_change_threshold else "white"
                axis.text(j, i, str(value), va="center", ha="center", color=color)

    # Plot
    plt.title(title)
    plt.yticks(range(matrix.shape[0]))
    plt.ylabel(rows_title)
    plt.gca().set_yticklabels(rows_labels)
    plt.xticks(range(matrix.shape[1]))
    plt.xlabel(cols_title)
    plt.gca().set_xticklabels(cols_labels)
    plt.tight_layout()
    plt.show()

    # Save
    if file_name is not None:
        #plt.savefig(file_name)
        figure.savefig(file_name, bbox_inches="tight")


def compute_graph_spectrogram(graph, signal, window_kernel, permutation=None):
    # We localize the window everywhere and report the frequencies
    if permutation is None: permutation = range(graph.N)
    spectrogram = np.zeros((graph.N, graph.N))
    for i in range(len(permutation)):
        window = window_kernel.localize(i)
        windowed_signal = window * signal
        spectrogram[:, permutation[i]] = graph.gft(windowed_signal) ** 2
    return spectrogram


"""
    Returns the neighbors of a vertex in a graph.
    --
    In:
        * graph: PyGSP graph.
        * vertex: Vertex for which to get neighbors.
    Out:
        * neighbors: A list of neighbors of the give vertex.
"""


def get_neighbors(graph, vertex):
    # We get the neighbors in the graph
    neighbors = list(graph.W[vertex].nonzero()[1])
    return neighbors


def basic_groups(graph):
    return np.array([graph.N // 10] * (graph.N // 3) +
                      [graph.N // 4] * (graph.N // 3) +
                      [graph.N // 2] * (graph.N - 2 * (graph.N // 3)))

def get_adjacency_matrix(graph):
    return [[graph.W[i, j] for i in range(graph.N)] for j in range(graph.N)]


def spectrogram_on_particular_case(graph, permutation, file_name=None):
    groups = basic_groups(graph)

    # We use a window defined by a heat kernel
    # Needs to be instanciated on a particular vertex to be the object we want

    kernel_scale = 10
    window_kernel = create_heat_kernel(graph, kernel_scale)
    # localized_kernel = window_kernel.localize(int(graph.N/2))
    x = np.array([graph.U[i, int(groups[i])] for i in range(graph.N)])
    x /= np.linalg.norm(x)
    # Plot
    plot_graph(graph, groups)
    spectrogram = compute_graph_spectrogram(graph, x, window_kernel, permutation=permutation)
    # Plot
    plot_matrix(spectrogram,
                cols_title="Vertex",
                cols_labels=range(graph.N),
                rows_title="Eigenvalue index",
                rows_labels=range(graph.N),
                title="Spectrogram",
                colorbar=True,
                file_name=file_name)
    return spectrogram


def create_groups_with_bfs(graph, nb_groups=3):
    N = graph.N
    weights = get_adjacency_matrix(graph)
    if not smb2.is_connected(weights):
        raise Exception("Error : the graph is not connected.")
    mult = 10 if N > nb_groups * 10 else N // nb_groups - 1
    nghbs = [get_neighbors(graph, i) for i in range(N)]
    # groups = [[] for _ in range(nb_groups)]
    explored_vertices = []
    groups = [0] * N
    for g in range(nb_groups):

        initial_vertex = np.random.randint(0, len(weights) - 1)
        while initial_vertex in explored_vertices:
            initial_vertex = np.random.randint(0, len(weights) - 1)
        vertices_to_explore = [initial_vertex]
        while len(explored_vertices) < (g + 1) * N // nb_groups + N % nb_groups and vertices_to_explore != []:
            current_vertex = vertices_to_explore.pop(0)
            explored_vertices.append(current_vertex)
            groups[current_vertex] = mult * (g + 1)
            vertices_to_explore += [n for n in nghbs[current_vertex] if
                                    n not in explored_vertices and n not in vertices_to_explore]

    while len(explored_vertices) < N:
        for i in range(N):
            if groups[i] == 0:
                for n in nghbs[i]:
                    if groups[n] != 0:
                        groups[i] = groups[n]
                        explored_vertices.append(i)
                        break

    return np.array(groups)


def spectrogram_with_groups(graph, groups, permutation, file_name=None, title='Spectrogramm', plot_spectro=False):
    kernel_scale = 10
    window_kernel = create_heat_kernel(graph, kernel_scale)
    x = np.array([graph.U[i, int(groups[i])] for i in range(graph.N)])
    x /= np.linalg.norm(x)
    spectrogram = compute_graph_spectrogram(graph, x, window_kernel, permutation=permutation)
    if plot_spectro:
        plot_graph(graph, x)
        plot_matrix(spectrogram,
                    cols_title="Vertex",
                    cols_labels=range(graph.N),
                    rows_title="Eigenvalue index",
                    rows_labels=range(graph.N),
                    title=title,
                    colorbar=True,
                    file_name=file_name)
    return spectrogram

def spectrogram_with_several_repartitions(graph, nb_repartitions, nb_groups=3, permutation=None, plot=False) :
    spectrograms = []
    for k in range(nb_repartitions) :
        groups = create_groups_with_bfs(graph, nb_groups)
        if plot:
            plot_graph(graph, groups)
        spectrogram = spectrogram_with_groups(graph, groups, permutation)
        spectrograms.append(spectrogram)
    return sum(spectrograms)

if __name__ == '__main__':
    from local_search import local_search

    Xsize = 250
    Ysize = 250
    # graph = create_gaussian_kernel_graph(100, Xsize=Xsize, Ysize=Ysize)
    graph = create_gaussian_kernel_graph(90, Xsize=Xsize, Ysize=Ysize)
    groups = create_groups_with_bfs(graph, nb_groups=3)
    print(groups.shape)
    print(groups)
    plot_graph(graph, groups, title='gaussian kernel graph with 3 groups got with BFS',
               file_name='gaussian_kernel_graph.png')


    weights = get_adjacency_matrix(graph)
    # best_permutation = smb2.spectral_sequencing(weights)
    best_permutation = smb2.mc_allister(weights)
    print(f'value of bandwith sum found  : {smb2.bandwidth_sum(best_permutation, weights)}')

    best_permutation, best_value = local_search(best_permutation, graph)
    print(best_permutation)
    print(f'value of bandwith sum found  : {smb2.bandwidth_sum(best_permutation, weights)}')

    print(best_permutation)
    spectrogram_with_groups(graph, groups, permutation=best_permutation,
                            file_name='spectrogram_gaussian_kernel_graph_groups_bfs.png',
                            title='mc_allister then local search on gaussian kernel graph with 3 groups got with BFS')






