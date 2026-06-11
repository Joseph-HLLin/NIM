import jax
import jax.numpy as jnp
from scipy.special import roots_jacobi, jacobi


def gauss_lobatto_jacobi_weights(Q):
    """
    Compute Gauss lobatto quadrature points and weights between [-1,1]

    --- Inputs: the number of quadrature points ---
    --- Outputs: the quadrature points and weights ---

    """
    X = roots_jacobi(Q - 2, 1, 1)[0]
    X = jnp.concatenate((jnp.array([-1]), jnp.array(X), jnp.array([1])))
    W0 = jacobi(Q - 1, 0, 0)(X)
    W = 2 / ((Q - 1) * Q * W0**2)

    return X, W
