"""
Tests on torch.clone (copy_hb_to_hb kernel)
03/18/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch

def test_torch_clone_1():
    x = torch.ones(10).hammerblade()
    y = x.clone()
    assert y.device == torch.device("hammerblade")
    assert x is not y
    assert torch.equal(x.cpu(), y.cpu())

def test_torch_clone_2():
    x = torch.randn(10).hammerblade()
    y = x.clone()
    assert y.device == torch.device("hammerblade")
    assert x is not y
    assert torch.equal(x.cpu(), y.cpu())

def test_torch_clone_3():
    x = torch.randn(3, 4).hammerblade()
    y = x.clone()
    assert y.device == torch.device("hammerblade")
    assert x is not y
    assert torch.equal(x.cpu(), y.cpu())
