import jax.numpy as jnp


def generate_model(bounds, nx_rkpm, ny_rkpm, n_subdomain, subdomain_size_factor=2.5):
    """
    Generate RKPM nodes, subdomain centers, and subdomain support radius.

    bounds = [x_min, x_max, y_min, y_max]
    n_subdomain is the number of subdomain centers in each direction.
    """
    length_x = bounds[1] - bounds[0]

    x_rkpm = jnp.linspace(bounds[0], bounds[1], nx_rkpm)
    y_rkpm = jnp.linspace(bounds[2], bounds[3], ny_rkpm)
    X, Y = jnp.meshgrid(x_rkpm, y_rkpm)
    rkpm_nodes = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    x_subdomain_center = jnp.linspace(bounds[0], bounds[1], n_subdomain)
    y_subdomain_center = jnp.linspace(bounds[2], bounds[3], n_subdomain)
    X, Y = jnp.meshgrid(x_subdomain_center, y_subdomain_center)
    centers = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    r_subdomain = length_x / (nx_rkpm - 1) * subdomain_size_factor

    return rkpm_nodes, centers, r_subdomain
