import torch

a = torch.rand([10, 9, 8, 7])
b = torch.rand([15, 9, 8, 7])

print(torch.matmul(a, b).size())
