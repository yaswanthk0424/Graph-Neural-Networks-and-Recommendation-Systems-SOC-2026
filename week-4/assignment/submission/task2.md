# Knowledge Graphs

## Introduction

A Knowledge Graph (KG) is a graph-based representation of structured knowledge in which entities are represented as nodes and relationships between entities are represented as edges. They are widely used in search engines, recommendation systems, question-answering systems, and biomedical applications.
---

## Components of a Knowledge Graph

### Entities

Entities are the objects, concepts, places, organizations, or people about which information is stored. In a Knowledge Graph, entities form the nodes of the graph.

Examples of entities include:

* IIT Bombay
* Mumbai
* India
* NVIDIA RTX GPUs
* NVIDIA
* Jensen Huang

### Relations

Relations describe how two entities are connected. Relations form the edges of the graph.

Examples of relations include:

* `located_in`
* `founded_by`
* `developed_by`
* `studies_at`

### Triples

Knowledge Graphs are commonly represented using triples of the form:

```text
(Subject, Relation, Object)
```

or formally😅:

```text
(Entity, Relation, Entity)
```

Examples:

```text
(IIT Bombay, located_in, Mumbai)

(Mumbai, located_in, India)

(NVIDIA RTX GPUs, developed_by, NVIDIA)
```

A Knowledge Graph can be viewed as a large collection of interconnected triples.

---

## Why Knowledge Graphs are Useful

Knowledge Graphs provide an intuitive way to represent relationships between entities. Unlike tabular databases, they naturally capture complex and interconnected information. This makes it easier to discover relationships and perform reasoning over data.

Knowledge Graphs also enable machines to understand context and connections between entities rather than treating information as isolated records.

---

## Applications

### Search Engines

Knowledge Graphs help search engines understand relationships between people, places, organizations, and events, enabling more accurate search results.

### Recommendation Systems

Users, products, movies, and books can be represented as entities, while interactions such as "likes" or "purchased" can be represented as relations.

### Question Answering Systems

Knowledge Graphs allow systems to answer questions by traversing relationships between entities.

### Biomedical Applications

Knowledge Graphs are used to represent relationships between diseases, drugs, genes, and proteins, helping researchers discover new insights.

---


