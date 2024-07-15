# Skeleton file for HW2 question 9
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.

FB_GRAPH_SIZE = 4039


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class UndirectedGraph:
    def __init__(self, number_of_nodes: int):
        """Assume that nodes are represented by indices/integers between 0 and (number_of_nodes - 1)."""
        self._nodes_num = number_of_nodes
        # graph: is a dictionary,
        # keys are the nodes,
        # values are set of nodes with edge to key node
        self._adjacency_set = {i: set() for i in range(number_of_nodes)}

    def add_edge(self, nodeA: int, nodeB: int):
        """ Adds an undirected edge to the graph, between nodeA and nodeB. Order of arguments should not matter"""
        if nodeA >= self._nodes_num or nodeB >= self._nodes_num:  # check if nodes exist in graph
            raise ValueError(
                f'At least one of the nodes is out of range. Received: ({nodeA}, {nodeB}). '
                f'Expected values are from 0 to {self._nodes_num}')
        if nodeA == nodeB:
            raise ValueError(f'Received the same node twice: {nodeA}')
        # add an edge
        self._adjacency_set[nodeA].add(nodeB)
        self._adjacency_set[nodeB].add(nodeA)

    def edges_from(self, nodeA: int):
        """ This method should return a list of all the nodes nodeB such that nodeA and nodeB are
        connected by an edge"""
        return list(self._adjacency_set[nodeA])

    def check_edge(self, nodeA: int, nodeB: int) -> bool:
        """ This method should return true is there is an edge between nodeA and nodeB, and false otherwise"""
        # note: since graph is undirected, if there exists (a,b), there must exist (b, a) - so it's enough to chck one
        return nodeA in self._adjacency_set[nodeB]

    def number_of_nodes(self) -> int:
        """ This method should return the number of nodes in the graph"""
        return self._nodes_num


def create_fb_graph(filename="facebook_combined.txt"):
    """ This method should return an undirected version of the
    facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes."""
    fb_G = UndirectedGraph(FB_GRAPH_SIZE)
    for line in open(filename):
        u, v = line.split(" ")
        fb_G.add_edge(int(u), int(v))
    return fb_G


# === Problem 9(a) ===

def contagion_brd(G, S, t):
    """Given an UndirectedGraph G, a list of adopters S (a list of integers in [0, G.number_of_nodes - 1]),
       and a float threshold t, perform BRD as follows:
       - Permanently infect the nodes in S with X
       - Infect the rest of the nodes with Y
       - Run BRD on the set of nodes not in S
       Return a list of all nodes infected with X after BRD converges."""
    # TODO: Implement this method
    pass


def q_completecascade_graph_fig4_1_left():
    """Return a float t s.t. the left graph in Figure 4.1 cascades completely."""
    # TODO: Implement this method 
    pass


def q_incompletecascade_graph_fig4_1_left():
    """Return a float t s.t. the left graph in Figure 4.1 does not cascade completely."""
    # TODO: Implement this method 
    pass


def q_completecascade_graph_fig4_1_right():
    """Return a float t s.t. the right graph in Figure 4.1 cascades completely."""
    # TODO: Implement this method 
    pass


def q_incompletecascade_graph_fig4_1_right():
    """Return a float t s.t. the right graph in Figure 4.1 does not cascade completely."""
    # TODO: Implement this method 
    pass


def main():
    # === Problem 9(b) === #
    # TODO: Put analysis code here
    # === Problem 9(c) === #
    # TODO: Put analysis code here
    # === OPTIONAL: Bonus Question 2 === #
    # TODO: Put analysis code here
    pass


# === OPTIONAL: Bonus Question 2 === #
def min_early_adopters(G, q):
    """Given an undirected graph G, and float threshold t, approximate the
       smallest number of early adopters that will call a complete cascade.
       Return an integer between [0, G.number_of_nodes()]"""
    pass


if __name__ == "__main__":
    main()
