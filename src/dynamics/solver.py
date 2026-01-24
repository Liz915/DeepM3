import torch

class ODESolver(object):
    def __init__(self, func):
        self.func = func

    def rk4_step(self, h, t, dt):
        """
        Runge-Kutta 4 阶积分法 (精度最高)
        h: 当前状态 [Batch, Hidden]
        t: 当前时间 [Batch, 1]
        dt: 时间步长 [Batch, 1]
        """
        # k1 = f(t, h)
        k1 = self.func(t, h)
        
        # k2 = f(t + 0.5*dt, h + 0.5*dt*k1)
        k2 = self.func(t + 0.5 * dt, h + 0.5 * dt * k1)
        
        # k3 = f(t + 0.5*dt, h + 0.5*dt*k2)
        k3 = self.func(t + 0.5 * dt, h + 0.5 * dt * k2)
        
        # k4 = f(t + dt, h + dt*k3)
        k4 = self.func(t + dt, h + dt * k3)
        
        return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def euler_step(self, h, t, dt):
        """
        欧拉积分法 
        """
        dh = self.func(t, h)
        return h + dh * dt