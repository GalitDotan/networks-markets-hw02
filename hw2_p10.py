# Skeleton file for HW2 question 10
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================

import matplotlib.pyplot as plt
# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# and UndirectGraph from hw2_p9
# please contact us before sumission if you want another package approved.
import numpy as np

INFINITE_DISTANCE = -1


class Color:
    WHITE = 'WHITE'
    GRAY = 'GRAY'
    BLACK = 'BLACK'


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class WeightedDirectedGraph:
    def __init__(self, number_of_nodes):
        """Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1."""
        self._nodes_num = number_of_nodes
        # graph: is a dictionary,
        # keys are the nodes, 
        # values are set of nodes with edge to key node
        self._adjacency_set = {i: set() for i in range(number_of_nodes)}
        self.edges_weights = {}

    def set_edge(self, origin_node, destination_node, weight=1):
        """ Modifies the weight for the specified directed edge, from origin to destination node,
            with specified weight (an integer >= 0). If weight = 0, effectively removes the edge from 
            the graph. If edge previously wasn't in the graph, adds a new edge with specified weight."""
        node_key = (origin_node, destination_node)
        if node_key in self.edges_weights:
            if weight == 0:  # delete edge
                self.edges_weights.pop(node_key)
                self._adjacency_set[origin_node].discard(destination_node)
            else:
                self.edges_weights[node_key] = weight
        else:
            if weight > 0:  # add edge
                self.edges_weights[node_key] = weight
                self._adjacency_set[origin_node].add(destination_node)

    def edges_from(self, origin_node):
        """ This method should return a list of all the nodes destination_node such that there is
            a directed edge (origin_node, destination_node) in the graph (i.e. with weight > 0)."""
        return list(self._adjacency_set[origin_node])

    def get_edge(self, origin_node, destination_node):
        """ This method should return the weight (an integer > 0) 
            if there is an edge between origin_node and 
            destination_node, and 0 otherwise."""
        node_key = (origin_node, destination_node)
        if node_key in self.edges_weights:
            return self.edges_weights[node_key]
        return 0

    def number_of_nodes(self):
        """ This method should return the number of nodes in the graph"""
        return self._nodes_num


def path_exists(target: int, distance: dict[int, int]) -> bool:
    return distance[target] != INFINITE_DISTANCE


def parse_parents_to_path(source: int, target: int, parent: dict[int, int]):
    """
    Parse the shortest path from source to target the parent dictionary
    Assuming the path exists.
    """
    curr = target
    path = [curr]
    while curr != source:
        curr = parent[curr]
        path.append(curr)
    path.reverse()
    return path


def shortest_path(G: WeightedDirectedGraph, i: int, j: int) -> list[int] | None:
    """ Given an UndirectedGraph G and nodes i,j, output the shortest path between nodes i and j in G."""
    n = G.number_of_nodes()
    color: dict[int, str] = {}  # used to track which nodes were already visited
    distance: dict[int, int] = {}  # minimal distance from i to each node
    parent: dict[int, int | None] = {}  # parent of each node in the BFS

    for node in range(n):
        color[node] = Color.WHITE
        distance[node] = INFINITE_DISTANCE
        parent[node] = None

    color[i] = Color.GRAY
    distance[i] = 0

    queue: list[int] = [i]

    # ssearch the shortest path
    while len(queue) > 0:
        node1 = queue.pop(0)
        for node2 in G.edges_from(node1):
            if color[node2] == Color.WHITE:
                color[node2] = Color.GRAY
                distance[node2] = distance[node1] + 1
                parent[node2] = node1
                queue.append(node2)
        color[node1] = Color.BLACK

    if not path_exists(target=j, distance=distance):
        return
    return parse_parents_to_path(source=i, target=j, parent=parent)


def copy_graph(G):
    """copy graph G and return the copy"""
    n = G.number_of_nodes()
    copied_g = WeightedDirectedGraph(n)
    for node in range(n):
        edges = G.edges_from(node)
        for edge in edges:
            copied_g.set_edge(node, edge, G.get_edge(node, edge))
    return copied_g


def min_weight(G, path):
    """find the minimum edge weight in path"""
    min_w = G.get_edge(path[0], path[1])
    for i in range(1, len(path) - 1):
        temp = G.get_edge(path[i], path[i + 1])
        if temp < min_w:
            min_w = temp
    return min_w


def create_F(G, G_copy, s):
    """creat F graph for max flow"""
    F = copy_graph(G)
    v = 0
    for node in range(F.number_of_nodes()):
        edges = F.edges_from(node)
        for edge in edges:
            F.set_edge(node, edge, F.get_edge(node, edge) - G_copy.get_edge(node, edge))
    for edge in F.edges_from(s):
        v += F.get_edge(s, edge)
    return v, F


# === Problem 10(a) ===

def max_flow(G, s, t) -> tuple[int, WeightedDirectedGraph]:
    """Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G as capacities.
       Return a tuple (v, F) where v is the integer value of the flow, and F is a maximum flow
       for G, represented by another WeightedDirectedGraph where edge weights represent
       the final allocated flow along that edge."""
    graph_copy = copy_graph(G)
    path = shortest_path(graph_copy, s, t)
    while path:
        temp_flow = min_weight(graph_copy, path)
        for i in range(len(path) - 1):
            new_weight_f = graph_copy.get_edge(path[i], path[i + 1]) - temp_flow
            new_weight_b = graph_copy.get_edge(path[i + 1], path[i]) + temp_flow
            graph_copy.set_edge(path[i], path[i + 1], new_weight_f)
            graph_copy.set_edge(path[i + 1], path[i], new_weight_b)
        path = shortest_path(graph_copy, s, t)
    return create_F(G, graph_copy, s)


# === Problem 10(c) ===
def max_matching(n: int, m: int, C: list[list[int]]) -> list[int]:
    """Given n drivers, m riders, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if driver i (in 0...n-1) and rider j (in 0...m-1) are compatible.
    If driver i and rider j are incompatible, then C[i][j] = 0. 
    Return an n-element array M where M[i] = j if driver i is matched with rider j,
    and M[i] = None if driver i is not matched."""
    graph = WeightedDirectedGraph(n + m + 2)
    matching = [0] * n
    s = n + m
    t = n + m + 1
    for i in range(n):
        for j in range(m):
            graph.set_edge(i, n + j, C[i][j])
    for i in range(n):
        graph.set_edge(s, i, 1)
    for j in range(m):
        graph.set_edge(n + j, t, 1)
    v, F = max_flow(graph, s, t)
    for i in range(n):
        j = F.edges_from(i)
        if len(j) == 0:
            j = None
        else:
            j = j[0] - n
        matching[i] = j
    return matching


# === Problem 10(d) ===
def random_driver_rider_bipartite_graph(n: int, p: float) -> list[list[int]]:
    """Returns an n x n constraints array C as defined for max_matching, representing a bipartite
       graph with 2n nodes, where each vertex in the left half is connected to any given vertex in the 
       right half with probability p."""
    graph = []
    for i in range(n):
        row = []
        for j in range(n):
            val = 1 if np.random.rand() < p else 0
            row.append(val)
        graph.append(row)
    return graph


def perfect_match_exist(n: int, p: float) -> bool:
    bipartite_graph = random_driver_rider_bipartite_graph(n, p)
    max_match = max_matching(n, n, bipartite_graph)
    return None not in max_match  # if exists node that is unmatched


def bonus_question(p_lambda):
    for n in range(50, 500, 50):
        p = p_lambda(n)  # math.log(n) / n
        cnt_success = 0
        cnt_tries = 0
        for _ in range(10):
            cnt_tries += 1
            if perfect_match_exist(n, p):
                cnt_success += 1
                # print(f'Success for ({n}, {p})')
            else:
                pass
                # print(f'Fail for ({n}, {p})')
        print(f'Result: {cnt_success / cnt_tries}')
        # assert cnt_success / cnt_tries >= 0.99


def main():
    x = []
    y = []
    n = 100
    for p in range(1, 100):
        x.append(p / 100)
        check = 0
        for i in range(10):
            if not perfect_match_exist(n, p / 100):  # if exists node that is unmatched
                continue
            check += 1
        y.append(check / 10)
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('probability of full match')
    plt.show()


if __name__ == "__main__":
    main()
    # bonus_question(p_lambda=lambda n: math.log(n) / n)
    # bonus_question(p_lambda=lambda n: (math.log(n) + 5) / n)
