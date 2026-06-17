#       A
#      / \
#     B   C
#    /     \
#   D       E

import numpy as np
import torch
A = torch.tensor([[1,1,1,0,0],[1,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[0,0,1,0,1]]) #self loop included
X = torch.tensor([[1,10],[2,8],[3,7],[4,6],[5,5]])

new_X = A@X
print(new_X)
