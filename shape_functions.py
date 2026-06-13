import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from weight_functions import cubic_b_spline_single


def p_vector(xi, xj):
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]

    return jnp.array([1.0, dx, dy, dx * dx, dx * dy, dy * dy])


def dp_vector(xi, xj):
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]

    return jnp.array([[0, -1, 0, -2 * dx, -dy, 0], [0, 0, -1, 0, -dx, -2 * dy]])


def inside_supported_domain(point, rkpm_nodes, support_radius):
    distances = jnp.linalg.norm(point - rkpm_nodes, axis=1)
    return distances <= support_radius


def get_max_qp_neighbors(subdomains, rkpm_nodes, r_subdomain):
    all_quad_points_flat = []

    for sub_data in subdomains:
        all_quad_points_flat.append(sub_data[0])

    all_quad_points_concatenated = jnp.concatenate(all_quad_points_flat, axis=0)

    all_neighbor_bools_for_max_calc = jax.vmap(
        inside_supported_domain, in_axes=(0, None, None)
    )(all_quad_points_concatenated, rkpm_nodes, r_subdomain)

    num_neighbors_per_qp = jnp.sum(all_neighbor_bools_for_max_calc, axis=1)

    max_neighbors = (
        jnp.max(num_neighbors_per_qp).item() if num_neighbors_per_qp.size > 0 else 1
    )

    return max_neighbors


@partial(jit, static_argnums=(3,))
def get_padded_neighbors_and_mask(
    single_quad_point, all_rkpm_nodes, support_radius, max_neighbors
):
    is_neighbor_bools = inside_supported_domain(
        single_quad_point, all_rkpm_nodes, support_radius
    )

    current_neighbor_ids_unpadded = jnp.where(
        is_neighbor_bools, size=max_neighbors, fill_value=len(all_rkpm_nodes)
    )[0]

    num_current_neighbors = jnp.sum(is_neighbor_bools)

    mask = jnp.arange(max_neighbors) < num_current_neighbors

    return current_neighbor_ids_unpadded, mask


@partial(jit, static_argnums=(5,))
def rkpm_shape_function_vmapped(
    quad_point, padded_neighbor_ids, mask, support_size, all_rkpm_nodes, max_neighbors
):
    m = 6

    # Append a dummy node to rkpm_nodes to safely index padded entries
    safe_rkpm_nodes = jnp.vstack([all_rkpm_nodes, jnp.zeros(all_rkpm_nodes.shape[-1])])

    # Get relevant rkpm nodes (including those corresponding to padded_neighbor_ids)
    rk_nodes_for_calc = safe_rkpm_nodes[padded_neighbor_ids]

    # Vmap cubic_b_spline_single to get weights and derivatives for all neighbors (padded and real)
    vmap_cubic_b_spline_for_rkpm = jax.vmap(
        cubic_b_spline_single, in_axes=(None, 0, None)
    )
    weights_all, dwdxs_all, dwdys_all = vmap_cubic_b_spline_for_rkpm(
        quad_point, rk_nodes_for_calc, support_size
    )

    # Apply mask to zero out contributions from padded neighbors
    weights_all_masked = weights_all * mask
    dwdxs_all_masked = dwdxs_all * mask
    dwdys_all_masked = dwdys_all * mask

    # Compute M and dM terms using vectorized operations and then sum
    p_vecs = jax.vmap(p_vector, in_axes=(None, 0))(quad_point, rk_nodes_for_calc)
    dp_vecs = jax.vmap(dp_vector, in_axes=(None, 0))(quad_point, rk_nodes_for_calc)

    # Calculate M
    outer_products_p_vec = jax.vmap(lambda p: jnp.outer(p, p))(p_vecs)
    M_terms = weights_all_masked[:, None, None] * outer_products_p_vec
    M = jnp.sum(M_terms, axis=0)  # (m, m)

    # Calculate dM
    dM_terms = jnp.zeros((max_neighbors, 2, m, m))
    for k in range(2):
        dw_k = jnp.array([dwdxs_all_masked, dwdys_all_masked])[k]
        dM_k_update_term1 = dw_k[:, None, None] * outer_products_p_vec
        dM_k_update_term2 = weights_all_masked[:, None, None] * jax.vmap(
            lambda dp, p: jnp.outer(dp, p), in_axes=(0, 0)
        )(dp_vecs[:, k, :], p_vecs)
        dM_k_update_term3 = weights_all_masked[:, None, None] * jax.vmap(
            lambda p, dp: jnp.outer(p, dp), in_axes=(0, 0)
        )(p_vecs, dp_vecs[:, k, :])
        dM_terms = dM_terms.at[:, k].set(
            dM_k_update_term1 + dM_k_update_term2 + dM_k_update_term3
        )

    dM = jnp.sum(dM_terms, axis=0)  # (2, m, m)

    # Handle potential singularity if M is zero due to no active neighbors
    # Add a small identity matrix to M for regularization if needed, or handle with lax.cond
    M_safe = M + jnp.where(
        jnp.all(M == 0), 1e-10 * jnp.eye(m), 0.0
    )  # Add epsilon if M is all zeros
    Minv = jnp.linalg.inv(M_safe)

    dMinv = jnp.zeros((2, m, m))
    for k in range(2):
        dMinv = dMinv.at[k].set(-Minv @ dM[k] @ Minv)

    px = jnp.array([1, 0, 0, 0, 0, 0])

    # Calculate N_output
    # Ensure weights_all is used for actual calculation, and mask at the end
    N_output_unmasked = jax.vmap(lambda w, pv: px @ Minv @ (w * pv), in_axes=(0, 0))(
        weights_all, p_vecs
    )
    N_output = N_output_unmasked * mask

    # Calculate dN_output
    dN_output_unmasked = jnp.zeros((max_neighbors, 2))
    for k in range(2):
        dw_k = jnp.array([dwdxs_all, dwdys_all])[
            k
        ]  # Use unmasked derivatives for calculation
        grad_k_terms = jax.vmap(
            lambda w, dw_k_val, pv, dpvk: (
                px @ Minv @ (w * dpvk)
                + px @ dMinv[k] @ (w * pv)
                + px @ Minv @ (dw_k_val * pv)
            ),
            in_axes=(0, 0, 0, 0),
        )(weights_all, dw_k, p_vecs, dp_vecs[:, k, :])
        dN_output_unmasked = dN_output_unmasked.at[:, k].set(grad_k_terms)

    dN_output = (
        dN_output_unmasked * mask[:, None]
    )  # Apply mask to both gradient components

    return N_output, dN_output


@partial(jit, static_argnums=(3,))
def calculate_shape_functions_vmapped_for_subdomain(
    quad_points_batch, rkpm_nodes_global, r_subdomain_global, max_neighbors
):
    # Vmap the get_padded_neighbors_and_mask function over the quad_points_batch
    batched_padded_neighbors_and_mask = jax.vmap(
        get_padded_neighbors_and_mask, in_axes=(0, None, None, None)
    )(quad_points_batch, rkpm_nodes_global, r_subdomain_global, max_neighbors)

    padded_neighbor_ids_batch = batched_padded_neighbors_and_mask[0]
    mask_batch = batched_padded_neighbors_and_mask[1]

    # Vmap the rkpm_shape_function_vmapped over the quad_points_batch
    N_batch, dN_batch = jax.vmap(
        rkpm_shape_function_vmapped,
        in_axes=(
            0,
            0,
            0,
            None,
            None,
            None,
        ),  # Map over quad_point, padded_neighbor_ids, mask
    )(
        quad_points_batch,
        padded_neighbor_ids_batch,
        mask_batch,
        r_subdomain_global,
        rkpm_nodes_global,
        max_neighbors,
    )

    return N_batch, dN_batch


def precompute_shape_functions(subdomain, rkpm_nodes, r_subdomain, max_neighbors):
    phi_all = []
    dphi_all = []

    for i_subdomain in range(len(subdomain)):
        current_subdomain_quad_coords = subdomain[i_subdomain][0]

        # Call the vmapped function for the current subdomain's quadrature points
        N_subdomain_batch, dN_subdomain_batch = (
            calculate_shape_functions_vmapped_for_subdomain(
                current_subdomain_quad_coords, rkpm_nodes, r_subdomain, max_neighbors
            )
        )

        phi_all.append(N_subdomain_batch)
        dphi_all.append(dN_subdomain_batch)

    return phi_all, dphi_all
