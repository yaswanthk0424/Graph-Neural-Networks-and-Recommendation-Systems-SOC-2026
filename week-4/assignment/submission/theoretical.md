# Comparison Between My Custom Graph Class and PyTorch Geometric (PyG)

## My Custom Graph Class

The custom graph class stores:

* Nodes
* Edges
* Node Features
* Edge Attributes (e.g., weights)

Example structure:

```python
class Graph:
    def __init__(self):
        self.Edges = [] #list
        self.Nodes = {} #dictionary
    def add_edge(self,u,v,weight):
        self.Edges.append((u,v,weight))#edges should be associated with weights
    def add_node(self,node_label,features):
        self.Nodes[node_label] = {"features":features}#nodes are associated with features
```

Nodes are stored in a dictionary along with their features, while edges are stored as a list of tuples containing the source node, destination node, and edge weight.

---

## PyTorch Geometric (PyG)

PyG represents graphs using the `Data` object.

Example:

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,
    edge_index=edge_list,
    edge_attr=edge_features,
    y=labels
)
```

Where:

* `x` stores node features
* `edge_index` stores graph connectivity
* `edge_attr` stores edge features
* `y` stores labels for training tasks

---

## Comparison

*PyG* represents the same information in a more efficient or optimised manner using tensors. Node features are stored in the `x` tensor, graph connectivity is stored in `edge_index`, and edge attributes can be stored in `edge_attr`. This tensor-based representation makes PyG efficient for GPU acceleration.

---

