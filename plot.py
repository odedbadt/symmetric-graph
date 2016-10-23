from matplotlib import pyplot as plt

def plot(graph, P):
    from itertools import chain
    edges = set(chain(*[[(i, j) if i <= j else (j, i) for j in neighbors] for i, neighbors in graph.items()]))
    plt.axis()
    for i, j in edges:
        plt.plot(P[[i, j], 0], P[[i, j], 1], 'k')
    plt.show()
