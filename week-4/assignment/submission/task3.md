# Over-Smoothing vs Over-Squashing

---

## What are they?

### Over-Smoothing

Over-smoothing is the phenomenon where node representations become increasingly similar as the number of message-passing layers increases. Since each layer aggregates information from neighboring nodes, repeated aggregation causes node embeddings to converge toward a common representation.

As a result, node embeddings become difficult to distinguish from one another, reducing the performance tasks such as node classification.

**Analogy:**

Suppose there are 100 crows with different opinions(on what?)
Everyday, each crow talks to its friends(only) and updates its opinions by averaging the opinions of nearby friends.
After a few days every crow thinks the same way
Individual view points disappear

### Over-Squashing

Over-squashing refers to the compression of information from a rapidly growing neighborhood into a fixed-size node embedding. As information travels through the graph, many distant messages must pass through a small number of nodes or edges, creating communication bottlenecks.

**Analogy:**

In a CS department, there may be 200 students with different problems or concerns, but only two Class Representatives (CRs) communicate with the professors. Information from all 200 students must pass through these two CRs, causing some information to be compressed or lost. Similarly, in a graph, information from many distant nodes may need to pass through only a few intermediate nodes before reaching the target node.

---

## Why do they happen?

### Why Over-Smoothing Happens

Consider a GCN:

```python
h = x

for _ in range(10):
    h = GCNConv(.................)(h, edge_index)
```

Each GCN layer performs:

```python
h = Aggregate(Neighbors(h))
```

Conceptually:

```text
Layer 0:
A = [1,0]
B = [0,1]

Layer 1:
A = avg(A,B)
B = avg(A,B)

Layer 2:
A = avg(A,B)
B = avg(A,B)

and so on 

Layer 10:
A ≈ B                 (Alt+247 for ≈)
```

Repeated neighborhood aggregation causes node embeddings to become increasingly similar, leading to over-smoothing.

### Why Over-Squashing Happens

Consider a deep GCN:

```python
h = x

for _ in range(10):
    h = GCNConv(..........)(h, edge_index)
```

After each layer, a node receives information from nodes that are one hop farther away.

```text
1 layer  → information from 1-hop neighbors             (Alt+26 gives →)
2 layers → information from 2-hop neighbors             (Alt+25 gives ↓)
5 layers → information from 5-hop neighbors
10 layers → information from a huge neighborhood
```

However, the embedding size remains fixed:

```python
GCNConv(128, 128)
```

So information from many distant nodes must be compressed into a single 128-dimensional vector.

This compression effect is known as over-squashing.

---

## How can we solve them?

### Solutions for Over-Smoothing

* Residual (skip) connections
* Using fewer message-passing layers
* Normalization techniques
* Architectures such as GraphSAGE and GAT

### Solutions for Over-Squashing

* Graph reconstructing to reduce bottlenecks
* Increasing embedding dimensions
* Improved aggregation and sampling strategies

---

## Tension Between Large Receptive Fields and Node Distinctiveness

Adding more GNN layers increases the receptive field of a node, allowing it to access information from increasingly distant parts of the graph.

```text
1 layer  → immediate neighbors
2 layers → neighbors of neighbors
3 layers → larger neighborhood
```

This additional information can improve learning. However, increasing the number of layers also increases the risk of over-smoothing and over-squashing.

Modern GNN research focuses on balancing these competing objectives by allowing information to propagate over long distances while preserving node-specific information.


