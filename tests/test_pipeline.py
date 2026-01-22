import pytest
import torch
from src.dynamics.solver import ODESolver
from src.dynamics.func import ODEFunc

def test_rk4_gradients():
    """
    验证手写的 RK4 求解器是否支持梯度反向传播
    """
    dim = 16
    func = ODEFunc(dim)
    solver = ODESolver(func)
    
    z0 = torch.randn(1, dim, requires_grad=True)
    t0 = torch.tensor(0.0)
    dt = torch.tensor(0.1)
    
    z1 = solver.rk4_step(z0, t0, dt)
    loss = z1.sum()
    loss.backward()
    
    assert z0.grad is not None, "Gradient to input z0 is broken"
    assert func.net[0].weight.grad is not None, "Gradient to ODE func parameters is broken"

def test_ode_integration():
    """
    验证积分数值稳定性
    """
    # y' = y, y(0)=1 => y(t) = e^t
    # 模拟一个简单的线性系统
    class SimpleFunc(torch.nn.Module):
        def forward(self, t, y):
            return y
            
    solver = ODESolver(SimpleFunc())
    z0 = torch.tensor([[1.0]])
    dt = torch.tensor(1.0)
    
    z1 = solver.rk4_step(z0, 0, dt)
    # e^1 = 2.71828
    # RK4 应该非常接近
    assert abs(z1.item() - 2.718) < 0.01, f"RK4 numerical error too large: {z1.item()}"