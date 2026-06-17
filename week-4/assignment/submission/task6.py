import networkx as nx
import numpy as np
#       A
#      / \
#     B   C
#    /     \
#   D       E

G = nx.Graph()
G.add_edges_from([("A","B"),("A","C"),("B","D"),("C","E")]) # graph done
# adjacency matrix A
A = nx.to_numpy_array(G)
A_modified = A + np.eye(len(G))
# edges list
E = list(G.edges())
print(A)
print(A_modified)
print(E)
