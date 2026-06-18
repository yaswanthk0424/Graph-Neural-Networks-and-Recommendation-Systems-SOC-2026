from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
Y = iris.target
print(X.shape) # (150,4) 150 flowers and 4 features per flower
print(Y.shape) # (150,) answers kind of
 
import networkx as nx
import numpy as np

G = nx.Graph()
for i in range(len(X)):
    G.add_node(i,features=X[i],label=int(Y[i])) #int() is needed
    # we actally store the label
    # our GNN uses this only at the step of classification
    
print(G.nodes[0]) # details of 1st node

threshold = 0.8
for i in range(len(X)):
    for j in range(i+1,len(X)):
        distance = np.linalg.norm(X[i]-X[j]) # st line distance (root of (x1-x2)^2 + (y1-y2)^2.....and rem things)
        
        if distance < threshold: # to not draw edge btw very unrelated nodes(features very diverging) 
            G.add_edge(i,j)
            
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print(list(G.edges())[:10])

#PyG needs tensors....
import torch
#PyG usually stores directed edge list
edges = []
for u,v in G.edges:
    edges.append([u,v])
    edges.append([v,u])
edge_index = torch.tensor(edges).t().contiguous()

# we need Tensor
x = torch.tensor(X,dtype=torch.float)
labels = torch.tensor(Y,dtype=torch.long)
print(labels.shape)

from torch_geometric.data import Data
data = Data(x=x,egde_index=edge_index,y=labels)
print(data)
