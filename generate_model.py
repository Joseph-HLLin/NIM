import jax.numpy as jnp


def generate_plate(bounds, nx_rkpm, ny_rkpm, n_subdomain, subdomain_size_factor=2.5):
    """
    Generate RKPM nodes, subdomain centers, and subdomain support radius.

    bounds = [x_min, x_max, y_min, y_max]
    n_subdomain is the number of subdomain centers in each direction.
    """
    length_x = bounds[1] - bounds[0]
    length_y = bounds[3] - bounds[2]
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


def generate_plate_w_hole(
    bounds, nx_rkpm, ny_rkpm, r_hole, n_subdomain, subdomain_size_factor=2.5
):
    """
    Generate RKPM nodes for the whole plate without subdomains.

    bounds = [x_min, x_max, y_min, y_max]
    """

    def is_inside_hole(x, y, hole_radius):
        return (x - 0) ** 2 + (y - 0) ** 2 < hole_radius**2

    length_x = bounds[1] - bounds[0]
    length_y = bounds[3] - bounds[2]

    x_pts = jnp.linspace(bounds[0], bounds[1], nx_rkpm)
    y_pts = jnp.linspace(bounds[2], bounds[3], ny_rkpm)
    X, Y = jnp.meshgrid(x_pts, y_pts)

    x_rkpm = []
    y_rkpm = []

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if is_inside_hole(X[i, j], Y[i, j], r_hole):
                x_rkpm.append(X[i, j])
                y_rkpm.append(Y[i, j])

    rkpm_nodes = jnp.stack([jnp.array(x_rkpm), jnp.array(y_rkpm)], axis=-1)

    x_pts = jnp.linspace(bounds[0], bounds[1], n_subdomain)
    y_pts = jnp.linspace(bounds[2], bounds[3], n_subdomain)
    X, Y = jnp.meshgrid(x_pts, y_pts)
    centers = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    r_subdomain = length_x / (nx_rkpm - 1) * subdomain_size_factor

    return rkpm_nodes, centers, r_subdomain
