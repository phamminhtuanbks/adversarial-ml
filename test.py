import torch
import torch.nn as nn

a = torch.rand((3,3), requires_grad=True)
epsilon = 2./20
b = a.data.clamp_(-epsilon, epsilon)
print(b)