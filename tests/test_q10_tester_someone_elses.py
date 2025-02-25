import unittest

from hw2_p10 import WeightedDirectedGraph, max_flow, max_matching, random_driver_rider_bipartite_graph, shortest_path


def _sorted_with_none(lst: list[int | None]):
    return sorted(lst, key=lambda x: (x is None, x))


class TestWeightedDirectedGraph(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedDirectedGraph(5)

    def test_initialization(self):
        self.assertEqual(self.graph.num_nodes, 5)
        self.assertEqual(self.graph.adj_list, {})

    def test_set_edge_add(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertIn(1, self.graph.edges_from(0))

    def test_set_edge_modify(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 5)
        self.assertEqual(self.graph.get_edge(0, 1), 5)

    def test_set_edge_remove(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.get_edge(0, 1), 0)
        self.assertNotIn(1, self.graph.edges_from(0))

    def test_edges_from(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 2, 4)
        self.assertEqual(self.graph.edges_from(0), [1, 2])
        self.assertEqual(self.graph.edges_from(1), [])
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.edges_from(0), [2])

    def test_get_edge(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertEqual(self.graph.get_edge(0, 2), 0)

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)


class TestWeightedDirectedGraph(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedDirectedGraph(5)

    def test_initialization(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)
        for i in range(self.graph.number_of_nodes()):
            self.assertEqual(self.graph.edges_from(i), [])

    def test_set_edge_add(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertIn(1, self.graph.edges_from(0))

    def test_set_edge_modify(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 5)
        self.assertEqual(self.graph.get_edge(0, 1), 5)

    def test_set_edge_remove(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.get_edge(0, 1), 0)
        self.assertNotIn(1, self.graph.edges_from(0))

    def test_edges_from(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 2, 4)
        self.assertEqual(self.graph.edges_from(0), [1, 2])
        self.assertEqual(self.graph.edges_from(1), [])
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.edges_from(0), [2])

    def test_get_edge(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertEqual(self.graph.get_edge(0, 2), 0)

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)


class TestWeightedDirectedGraphBFS(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedDirectedGraph(6)
        # Construct a graph for testing
        self.graph.set_edge(0, 1, 1)
        self.graph.set_edge(0, 2, 1)
        self.graph.set_edge(1, 3, 1)
        self.graph.set_edge(2, 4, 1)
        self.graph.set_edge(3, 5, 1)
        self.graph.set_edge(4, 5, 1)

    def test_bfs_path_exists(self):
        parent = [-1] * self.graph.number_of_nodes()
        self.assertTrue(shortest_path(self.graph, 0, 5))

    def test_bfs_no_path(self):
        parent = [-1] * self.graph.number_of_nodes()
        self.assertFalse(shortest_path(self.graph, 1, 4))
        self.assertEqual(parent[4], -1)  # Ensure the parent array is not wrongly updated

    def test_bfs_single_node(self):
        single_node_graph = WeightedDirectedGraph(1)
        parent = [-1] * single_node_graph.number_of_nodes()
        self.assertEqual(shortest_path(single_node_graph, 0, 0), [0])

    def test_bfs_disconnected_graph(self):
        self.graph.set_edge(4, 5, 0)  # Remove edge to disconnect part of the graph
        parent = [-1] * self.graph.number_of_nodes()
        self.assertFalse(shortest_path(self.graph, 4, 5))
        self.assertEqual(parent[5], -1)


class TestMaxFlow(unittest.TestCase):
    def test_small_graph(self):
        G = WeightedDirectedGraph(4)
        G.set_edge(0, 1, 1)
        G.set_edge(0, 2, 3)
        G.set_edge(1, 2, 2)
        G.set_edge(1, 3, 1)
        G.set_edge(2, 3, 1)

        s, t = 0, 3
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 2)

    def test_medium_graph(self):
        G = WeightedDirectedGraph(6)
        G.set_edge(0, 1, 16)
        G.set_edge(0, 2, 13)
        G.set_edge(1, 2, 10)
        G.set_edge(1, 3, 12)
        G.set_edge(2, 1, 4)
        G.set_edge(2, 4, 14)
        G.set_edge(3, 2, 9)
        G.set_edge(3, 5, 20)
        G.set_edge(4, 3, 7)
        G.set_edge(4, 5, 4)

        s, t = 0, 5
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 23)

        expected_flows = {
            (0, 1): 12,
            (1, 0): -12,
            (0, 2): 11,
            (2, 0): -11,
            (1, 2): 0,
            (2, 1): 0,
            (1, 3): 12,
            (3, 1): -12,
            (2, 4): 11,
            (4, 2): -11,
            (3, 2): 1,
            (2, 3): -1,
            (3, 5): 19,
            (5, 3): -19,
            (4, 3): 7,
            (3, 4): -7,
            (4, 5): 4,
            (5, 4): -4,
        }

        for u in range(flow_graph.number_of_nodes()):
            for v in flow_graph.edges_from(u):
                self.assertEqual(flow_graph.get_edge(u, v), expected_flows.get((u, v), 0))

    def test_disconnected_graph(self):
        G = WeightedDirectedGraph(4)
        G.set_edge(0, 1, 3)
        G.set_edge(2, 3, 2)

        s, t = 0, 3
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 0)

    def test_no_edges(self):
        G = WeightedDirectedGraph(4)

        s, t = 0, 3
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 0)


class TestMaxMatching(unittest.TestCase):

    def test_basic_matching(self):
        # Test a basic case where all drivers and riders are compatible
        n = 3
        m = 2
        C = [
            [1, 1],
            [1, 1],
            [1, 1]
        ]
        expected_matching = [0, 1, None]  # Drivers 0 and 1 matched with riders 0 and 1 respectively

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_no_matching(self):
        # Test where no drivers and riders are compatible
        n = 3
        m = 2
        C = [
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        expected_matching = [None, None, None]  # No drivers matched with any riders

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_partial_matching(self):
        # Test where only some drivers and riders are compatible
        n = 3
        m = 2
        C = [
            [1, 0],
            [0, 1],
            [1, 0]
        ]
        expected_matching = [0, 1, None]  # Drivers 0 and 1 matched with riders 0 and 1 respectively

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_large_matching(self):
        # Test with a larger number of drivers and riders
        n = 5
        m = 4
        C = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ]
        expected_matching = [0, 1, 2, None, 3]  # Example expected matching

        result = max_matching(n, m, C)
        self.assertEqual(_sorted_with_none(result),
                         _sorted_with_none(expected_matching))

    def test_empty_input(self):
        # Test with empty input
        n = 0
        m = 0
        C = []
        expected_matching = []  # No drivers and riders

        result = max_matching(n, m, C)
        self.assertEqual(_sorted_with_none(result),
                         _sorted_with_none(expected_matching))

    def test_large_numbers(self):
        # Test with large numbers of drivers and riders
        n = 100
        m = 50
        # Assume all pairs are compatible for simplicity
        C = [[1] * m for _ in range(n)]
        expected_matching = list(range(m)) + [None] * (n - m)  # All drivers matched with all riders

        result = max_matching(n, m, C)
        self.assertEqual(_sorted_with_none(result),
                         _sorted_with_none(expected_matching))


class TestRandomDriverRiderBipartiteGraph(unittest.TestCase):

    def test_graph_generation(self):
        # Test the basic case with n = 3 and p = 0.5
        n = 3
        p = 0.5
        C = random_driver_rider_bipartite_graph(n, p)

        # Check dimensions
        self.assertEqual(len(C), n)
        for row in C:
            self.assertEqual(len(row), n)

        # Check values are either 0 or 1
        for row in C:
            for value in row:
                self.assertIn(value, [0, 1])

    def test_edge_cases(self):
        # Test edge cases: n = 0, p = 0 and p = 1
        n_values = [0, 1, 5, 10, 100]
        p_values = [0, 1]

        for n in n_values:
            for p in p_values:
                C = random_driver_rider_bipartite_graph(n, p)

                # Check dimensions
                self.assertEqual(len(C), n)
                for row in C:
                    self.assertEqual(len(row), n)

                # Check all values are either 0 or 1
                for row in C:
                    for value in row:
                        self.assertEqual(value, p)


if __name__ == '__main__':
    unittest.main()
