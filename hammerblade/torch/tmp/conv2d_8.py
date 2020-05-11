import torch
import torch.nn.functional as F

torch.manual_seed(42)

inputs = torch.rand(1, 1, 8, 8)
weights = torch.rand(1, 1, 3, 3)

out = F.conv2d(inputs, weights)
out_hb = F.conv2d(inputs.hammerblade(), weights.hammerblade())

assert torch.allclose(out, out_hb.cpu())
