import torch
import torch.nn.functional as F

torch.manual_seed(42)

inputs = torch.rand(1, 1, 8, 8).hammerblade()
weights = torch.rand(1, 1, 3, 3).hammerblade()

print(F.conv2d(inputs, weights))
