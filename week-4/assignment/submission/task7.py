class Graph:
    def __init__(self):
        self.Edges = [] #list
        self.Nodes = {} #dictionary
    def add_edge(self,u,v,weight):
        self.Edges.append((u,v,weight))#edges should be associated with weights
    def add_node(self,node_label,features):
        self.Nodes[node_label] = {"features":features}#nodes are associated with features
    def neighbours(self,node):
        # our Edge is like (u,v,weight) making life easy
        nbh = []
        for u,v,weight in self.Edges():
            if u==node:
                nbh.append(u)
            elif v==node:
                nbh.append(v)
        return nbh
    

            