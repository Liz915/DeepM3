import torch

class ODESolver(object):
    def __init__(self, func):
        self.func = func

    def _call_func(self, t, h, dt_ctx):
        try:
            return self.func(t, h, dt_ctx)
        except TypeError:
            # Backward compatibility for functions defined as f(t, h).
            return self.func(t, h)

    def rk4_step(self, h, t, dt, dt_ctx=None):
        """
        Runge-Kutta 4 integration step.
        h:      [B, D] hidden state
        t:      [B, 1] start time of this step
        dt:     [B, 1] integration step size
        dt_ctx: [B, 1] optional delta-time conditioning signal for ODEFunc
        """
        if dt_ctx is None:
            dt_ctx = dt

        k1 = self._call_func(t, h, dt_ctx)
        k2 = self._call_func(t + 0.5 * dt, h + 0.5 * dt * k1, dt_ctx)
        k3 = self._call_func(t + 0.5 * dt, h + 0.5 * dt * k2, dt_ctx)
        k4 = self._call_func(t + dt, h + dt * k3, dt_ctx)

        return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def euler_step(self, h, t, dt, dt_ctx=None):
        """
        Euler integration step.
        """
        if dt_ctx is None:
            dt_ctx = dt
        dh = self._call_func(t, h, dt_ctx)
        return h + dh * dt
