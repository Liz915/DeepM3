from src.dynamics.solver import ODESolver
import torch

# dummy derivative f
func = lambda t, z: z * 0 + 1  
ode = ODESolver(func)

t = torch.tensor([[1.0,1.0,2.0,4.0]])
z0 = torch.tensor([[0.0]])

out = ode(z0, t)
print(out)