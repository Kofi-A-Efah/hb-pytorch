import torch
import torch.nn.functional as F

torch.manual_seed(42)

x = torch.rand(8, 8)
y = torch.rand(8, 8)

out = torch.mm(x, y)
out_hb = torch.mm(x.hammerblade(), y.hammerblade())

assert torch.allclose(out, out_hb.cpu())
