import torch
import torch.nn.functional as F

torch.manual_seed(42)

x = torch.rand(8, 8).hammerblade()
y = torch.rand(8, 8).hammerblade()

print(torch.mm(x, y))
