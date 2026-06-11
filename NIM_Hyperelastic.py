import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

import jax
import jax.numpy as jnp
from jax import random, jit
from jax import config
from functools import partial
from jax.nn import relu, tanh
import jaxopt
from jax import grad

import matplotlib.pyplot as plt
from scipy.special import roots_jacobi, jacobi
import time
import sys

from quadrature import gauss_lobatto_jacobi_weights
from generate_model import generate_model
from weight_functions import cubic_b_spline_single

config.update("jax_enable_x64", False)
print(jax.devices())

# ============= model parameters =============#

length_x = 1
length_y = 1
nx_rkpm = 21
ny_rkpm = 21
n_subdomain = 41
subdomain_size_factor = 2.5

domain_bounds = jnp.array([0, 1, 0, 1])

rkpm_nodes, centers, r_subdomain = generate_model(
    domain_bounds,
    nx_rkpm=nx_rkpm,
    ny_rkpm=ny_rkpm,
    n_subdomain=n_subdomain,
    subdomain_size_factor=subdomain_size_factor,
)
num_rkpm_nodes = rkpm_nodes.shape[0]
num_subdomains = centers.shape[0]

print("//-------- Model Parameters--------//")
print(f"Number of RKPM nodes: {num_rkpm_nodes}")
print(f"Number of subdomains: {num_subdomains}")

# ============= plot Model =============#
plt.figure(figsize=(6, 6))
plt.scatter(rkpm_nodes[:, 0], rkpm_nodes[:, 1])

# ============= material parameters =============#
E = 1000
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

quadrature_points, quadrature_weights = gauss_lobatto_jacobi_weights(5)
quadrature_partition = 2

cubic_b_spline_single()
