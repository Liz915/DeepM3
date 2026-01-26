# System 1: Continuous-Time User Dynamics

This module implements the core **Neural ODE–based user dynamics model** in DeepM3.
User preference evolution is modeled as a continuous trajectory in latent space,
enabling principled handling of irregularly sampled interaction sequences.

The design supports configurable ODE solvers (Euler / RK4) and discrete ablations,
providing a unified framework for studying accuracy–latency trade-offs in production systems.