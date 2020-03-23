"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""

from __future__ import absolute_import
import torch
from hypothesis import assume, given, settings, HealthCheck
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

# test of adding two tensors

def test_elementwise_add_1():
    x1 = torch.ones(1, 10)
    x2 = torch.ones(1, 10)
    y = x1 + x2
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    y_h = x1_h + x2_h
    y_c = y_h.cpu()
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_c, y)

def test_elementwise_add_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    y = x1 + x2
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    y_h = x1_h + x2_h
    y_c = y_h.cpu()
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_c, y)

def test_elementwise_add_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    y = x1 + x2
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    y_h = x1_h + x2_h
    y_c = y_h.cpu()
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_c, y)

def test_elementwise_add_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    y = x1 + x2
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    y_h = x1_h + x2_h
    y_c = y_h.cpu()
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_c, y)

@given(inputs=hu.tensors(n=2))
def test_elementwise_add_hypothesis(inputs):
    def elementwise_add(inputs):
        assert len(inputs) == 2
        return inputs[0] + inputs[1]
    hu.assert_hb_checks(elementwise_add, inputs)

def test_elementwise_in_place_add():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    x1.add_(x2)
    x1_h.add_(x2_h)
    assert x1_h.device == torch.device("hammerblade")
    x1_h_c = x1_h.cpu()
    assert torch.equal(x1_h_c, x1)

def test_add_with_scalar():
    x = torch.rand(16)
    x_h = x.hammerblade()
    y = x + 5
    y_h = x_h + 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y)

def test_elementwise_sub_1():
    x = torch.ones(1, 10)
    y = torch.ones(1, 10)
    z = x - y
    z_h = x.hammerblade() - y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_sub_2():
    x = torch.ones(4, 5)
    y = torch.ones(4, 5)
    z = x - y
    z_h = x.hammerblade() - y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_sub_3():
    x = torch.rand(1, 128)
    y = torch.rand(1, 128)
    z = x - y
    z_h = x.hammerblade() - y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_sub_4():
    x = torch.rand(16, 32)
    y = torch.rand(16, 32)
    z = x - y
    z_h = x.hammerblade() - y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_in_place_sub():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    x1.sub_(x2)
    x1_h.sub_(x2_h)
    assert x1_h.device == torch.device("hammerblade")
    x1_h_c = x1_h.cpu()
    assert torch.equal(x1_h_c, x1)

def test_sub_with_scalar():
    x = torch.rand(16)
    x_h = x.hammerblade()
    y = x - 5
    y_h = x_h - 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y)
