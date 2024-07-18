import pytest

import hw2_p10
import hw2_p9


# DO NOT SUBMIT THIS FILE
# It is just an example of a few tests that we will run on your code that you can use as a starting point
# to make sure the code is correct.
# You should put the two python files in the same folder and run this one

def test_number_of_nodes():
    n5 = 5
    n0 = 0
    graph = hw2_p9.UndirectedGraph(n5)
    assert graph.number_of_nodes() == n5
    graph = hw2_p9.UndirectedGraph(n0)
    assert graph.number_of_nodes() == n0


def test_empty_graph():
    n = 10
    graph = hw2_p9.UndirectedGraph(n)
    assert graph.number_of_nodes() == n
    for i in range(n):
        for j in range(n):
            if i != j:
                assert not graph.check_edge(i, j)
        assert len(graph.edges_from(i)) == 0
    print("tested empty graph")


def test_edges_from_and_check_edge():
    n = 5
    graph = hw2_p9.UndirectedGraph(n)
    for i in range(n):
        assert len(graph.edges_from(i)) == 0
    for i in [0, 2, 3, 4]:
        graph.add_edge(1, i)
    for i in [0, 2, 3, 4]:
        assert len(graph.edges_from(i)) == 1
        assert graph.check_edge(i, 1)
        assert graph.check_edge(1, i)
    assert len(graph.edges_from(1)) == 4
    graph.add_edge(0, 4)
    assert len(graph.edges_from(0)) == 2
    assert len(graph.edges_from(4)) == 2
    assert len(graph.edges_from(2)) == 1
    assert len(graph.edges_from(3)) == 1
    assert len(graph.edges_from(1)) == 4
    assert not graph.check_edge(2, 3)
    assert not graph.check_edge(3, 4)
    assert not graph.check_edge(0, 3)
    print("tested edge_from function, check_edge function")


# @pytest.mark.parametrize("num_of_nodes", [10])
# def test_avg_shortest_path(num_of_nodes):
#    test_graph = hw2_p9.UndirectedGraph(3)
#    assert test_graph.number_of_nodes() == 3
#    test_graph.add_edge(0, 1)
#    test_graph.add_edge(2, 1)
# avg = hw2_p10.avg_shortest_path(test_graph)
# assert avg > 1.2  # Should be close to true value of 1.3333
# assert avg < 1.5  # Should be close to true value of 1.3333

#    half_n = int(num_of_nodes / 2)
#    test_graph = hw2_p9.UndirectedGraph(num_of_nodes)
#    for i in range(num_of_nodes - 1):
#        test_graph.add_edge(i, i + 1)
#    assert (hw2_p9.avg_shortest_path(test_graph) - (num_of_nodes + 1) / 3) < 0.5  # since it's just a sample
#    print("tested avg_shortest_path function")


@pytest.mark.parametrize("num_of_nodes", [10])
def test_shortest_path_empty(num_of_nodes: int):
    # empty graph
    test_graph = hw2_p10.WeightedDirectedGraph(num_of_nodes)
    for i in range(num_of_nodes - 1):
        for j in range(i + 1, num_of_nodes):
            assert not hw2_p10.shortest_path(test_graph, i, j)  # empty graph
    assert not hw2_p10.shortest_path(test_graph, num_of_nodes - 1, 0)  # empty graph


@pytest.mark.parametrize("num_of_nodes, source,target, expected_path", [(10, 3, 1, None),
                                                                        (10, 5, 0, None),
                                                                        (10, 0, 5, (0, 1, 2, 3, 4, 5)),
                                                                        (10, 0, 0, (0,)),
                                                                        (10, 0, 1, (0, 1)),
                                                                        (10, 2, 4, (2, 3, 4)),
                                                                        (10, 5, 0, None)])
def test_shortest_path_line(num_of_nodes: int, source: int, target: int, expected_path: tuple[int] | None):
    # line
    # The graph: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> ...
    test_graph = hw2_p10.WeightedDirectedGraph(num_of_nodes)

    for i in range(1, num_of_nodes):
        test_graph.set_edge(i - 1, i)

    if expected_path:
        expected_path = list(expected_path)

    assert hw2_p10.shortest_path(test_graph, source, target) == expected_path

    for i in range(num_of_nodes - 1):
        for j in range(num_of_nodes - 1):
            if i > j:
                assert hw2_p10.shortest_path(test_graph, i, j) is None
            else:
                assert len(hw2_p10.shortest_path(test_graph, i, j)) == j - i + 1


@pytest.mark.parametrize("num_of_nodes, source,target, expected_path", [(10, 3, 1, (3, 4, 5, 6, 7, 8, 9, 0, 1)),
                                                                        (10, 5, 0, (5, 6, 7, 8, 9, 0)),
                                                                        (10, 0, 5, (0, 1, 2, 3, 4, 5)),
                                                                        (10, 0, 0, (0,)),
                                                                        (10, 0, 1, (0, 1)),
                                                                        (10, 2, 4, (2, 3, 4)),
                                                                        (10, 9, 3, (9, 0, 1, 2, 3))])
def test_shortest_path_cycle(num_of_nodes: int, source: int, target: int, expected_path: tuple[int] | None):
    # line
    # The graph: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> ...
    test_graph = hw2_p10.WeightedDirectedGraph(num_of_nodes)

    for i in range(1, num_of_nodes):
        test_graph.set_edge(i - 1, i)
    test_graph.set_edge(num_of_nodes - 1, 0)

    if expected_path:
        expected_path = list(expected_path)

    assert hw2_p10.shortest_path(test_graph, source, target) == expected_path

    for i in range(num_of_nodes - 1):
        for j in range(num_of_nodes - 1):
            if i > j:
                assert len(hw2_p10.shortest_path(test_graph, i, j)) == num_of_nodes - i + j + 1
            else:
                assert len(hw2_p10.shortest_path(test_graph, i, j)) == j - i + 1


# @pytest.mark.parametrize("n,p", [(100, 0.5)])
# def test_create_graph_with_p(n: int, p: float):
#    graph = hw2_p10.create_graph(n, p)
#    outbound_node1_list = graph.edges_from(1)
#    # With very, very small probability this test will fail even for correct implementation
#    assert len(outbound_node1_list) > 2


def test_fb_graph():
    # Now, assuming that facebook_combined.txt is in the same directory as tester.py
    fb_graph = hw2_p9.create_fb_graph()
    assert fb_graph.number_of_nodes() == 4039
    assert fb_graph.check_edge(107, 1453)
    assert not fb_graph.check_edge(133, 800)
