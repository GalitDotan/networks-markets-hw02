from hw2_p10 import *


def test_path_exists():
    distance = {2: 1, 3: INFINITE_DISTANCE, 4: 0, 5: 5}
    assert path_exists(target=2, distance=distance)
    assert not path_exists(target=3, distance=distance)
    assert path_exists(target=4, distance=distance)
    assert path_exists(target=5, distance=distance)


def test_parse_parents_to_path():
    # graph: 1 -> 2 -> 3 -> 4 -> 5
    parent = {i + 1: i for i in range(1, 5)}
    assert parse_parents_to_path(1, 5, parent) == list(range(1, 6))
    parent[6] = 1
    parent[7] = 6
    assert parse_parents_to_path(1, 7, parent) == [1, 6, 7]


def test_shortest_path():
    graph = WeightedDirectedGraph(number_of_nodes=10)  # node: [0,..., 9]
    graph.set_edge(0, 1)
    graph.set_edge(0, 2)
    graph.set_edge(1, 3)
    graph.set_edge(1, 4)
    graph.set_edge(2, 5)
    graph.set_edge(4, 6)
    graph.set_edge(5, 7)
    graph.set_edge(5, 8)
    assert shortest_path(graph, 0, 0) == [0]
    assert shortest_path(graph, 0, 3) == [0, 1, 3]
    assert shortest_path(graph, 0, 4) == [0, 1, 4]
    assert shortest_path(graph, 0, 6) == [0, 1, 4, 6]
    assert shortest_path(graph, 0, 7) == [0, 2, 5, 7]
    assert shortest_path(graph, 0, 8) == [0, 2, 5, 8]
    assert shortest_path(graph, 1, 6) == [1, 4, 6]
    assert shortest_path(graph, 3, 6) is None
