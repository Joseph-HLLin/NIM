import jax
import jax.numpy as jnp
from jax import vmap, jit
from quadrature import gauss_lobatto_jacobi_weights

## This file contains the weight function and its derivatives for the RKPM method, as well as the construction of subdomains and their quadrature points, weights, and Jacobians.


def cubic_b_spline_single(xi, xj, support_radius):
    """
    Computes cubic B-spline weight and its derivatives for a single point.

    Args:
        xi (array): Current quadrature point (2D).
        xj (array): Center of the subdomain (2D).
        support_radius (float): Radius of influence for the B-spline.

    Returns:
        tuple: (weight, dweight_dx, dweight_dy) as scalar JAX arrays.
    """
    r_vec = xi - xj
    r_phys = jnp.linalg.norm(r_vec)
    r = r_phys / support_radius  # Normalized distance

    w = 0.0
    dwdr = 0.0

    # Conditions for cubic B-spline
    condition_r_le_half = r <= 0.5
    condition_r_gt_half_le_one = (0.5 < r) & (r <= 1.0)

    w = jnp.where(condition_r_le_half, (2.0 / 3) - 4 * r * r + 4 * r**3, w)
    w = jnp.where(
        condition_r_gt_half_le_one, (4.0 / 3) - 4 * r + 4 * r**2 - (4 / 3) * r**3, w
    )
    # If r > 1, w remains 0.0

    dwdr = jnp.where(condition_r_le_half, (-8 * r + 12 * r**2), dwdr)
    dwdr = jnp.where(condition_r_gt_half_le_one, -4 + 8 * r - 4 * r**2, dwdr)
    # If r > 1, dwdr remains 0.0

    # Avoid division by zero if r_phys is zero
    drdx = jnp.where(r_phys == 0.0, 0.0, r_vec[0] / (r_phys * support_radius))
    drdy = jnp.where(r_phys == 0.0, 0.0, r_vec[1] / (r_phys * support_radius))

    dwdx = dwdr * drdx
    dwdy = dwdr * drdy

    return w, dwdx, dwdy


@jit
def weight_function(quadrature_points, center, r_subdomain):
    """
    Computes weights and their derivatives for multiple quadrature points using jax.vmap.

    Args:
        quadrature_points (array): Array of quadrature points (N, 2).
        center (array): Center of the subdomain (2D).
        r_subdomain (float): Radius of influence for the B-spline.

    Returns:
        tuple: (weights, dweights_dx, dweights_dy) as 2D JAX arrays (N, 1).
    """
    # vmap cubic_b_spline_single over the first argument (quadrature_points)
    vmap_cubic_b_spline = vmap(cubic_b_spline_single, in_axes=(0, None, None))
    weights, dwdxs, dwdys = vmap_cubic_b_spline(quadrature_points, center, r_subdomain)
    return weights[:, None], dwdxs[:, None], dwdys[:, None]  # Return 2D arrays (N, 1)


def construct_subdomain_weights(
    subdomain_centers, domain_bounds, r_subdomain, subdomain_partition, quad_order
):
    Quad_coord, Quad_weight = gauss_lobatto_jacobi_weights(quad_order)

    # Reshape to (quad_order,) for easier meshgrid/outer product operations
    quad_coord_1d = Quad_coord.squeeze()
    quad_weight_1d = Quad_weight.squeeze()

    W_all, DWx_all, DWy_all = [], [], []

    subdomain = []

    for center in subdomain_centers:
        # Corrected bounding box calculation using jnp.maximum/jnp.minimum and r_subdomain
        bbox_xmin = jnp.maximum(center[0] - r_subdomain, domain_bounds[0])
        bbox_xmax = jnp.minimum(center[0] + r_subdomain, domain_bounds[1])
        bbox_ymin = jnp.maximum(center[1] - r_subdomain, domain_bounds[2])
        bbox_ymax = jnp.minimum(center[1] + r_subdomain, domain_bounds[3])

        subdomain_length_x = bbox_xmax - bbox_xmin
        subdomain_length_y = bbox_ymax - bbox_ymin

        # Lists to collect points, weights, jacobians for current subdomain across all partitions
        all_partition_physical_coords = []
        all_partition_weights = []
        all_partition_jacobians = []

        quad_domain_length_x = subdomain_length_x / subdomain_partition
        quad_domain_length_y = subdomain_length_y / subdomain_partition

        for i in range(subdomain_partition):
            for j in range(subdomain_partition):
                current_quad_domain_xmin = bbox_xmin + i * quad_domain_length_x
                current_quad_domain_ymin = bbox_ymin + j * quad_domain_length_y

                # Scale the Lobatto points to the current partition's domain (1D arrays)
                scaled_quad_x_coord = (
                    current_quad_domain_xmin
                    + 0.5 * quad_domain_length_x * (quad_coord_1d + 1)
                )
                scaled_quad_y_coord = (
                    current_quad_domain_ymin
                    + 0.5 * quad_domain_length_y * (quad_coord_1d + 1)
                )

                # Generate 2D grid of physical coordinates for the current partition
                x_grid, y_grid = jnp.meshgrid(scaled_quad_x_coord, scaled_quad_y_coord)
                partition_physical_coords = jnp.stack(
                    [x_grid.ravel(), y_grid.ravel()], axis=-1
                )
                all_partition_physical_coords.append(partition_physical_coords)

                # Generate 2D grid of weights for the current partition using outer product
                partition_weights_2d = quad_weight_1d[:, None] * quad_weight_1d[None, :]
                all_partition_weights.append(
                    partition_weights_2d.ravel()
                )  # Flatten to 1D

                # Calculate Jacobian for the current partition and create an array of its size
                partition_jacobian_value = (
                    0.25 * quad_domain_length_x * quad_domain_length_y
                )
                all_partition_jacobians.append(
                    jnp.full(
                        partition_physical_coords.shape[0], partition_jacobian_value
                    )
                )

        # Concatenate collected arrays for the current subdomain
        subdomain_all_quad_coords = jnp.concatenate(
            all_partition_physical_coords, axis=0
        )
        subdomain_all_quad_weights = jnp.concatenate(all_partition_weights, axis=0)
        subdomain_all_quad_jacobians = jnp.concatenate(all_partition_jacobians, axis=0)

        subdomain.append(
            [
                subdomain_all_quad_coords,
                subdomain_all_quad_weights,
                subdomain_all_quad_jacobians,
            ]
        )

        # Now call the vectorized weight function once for all points in this subdomain
        W, DWx, DWy = weight_function(subdomain_all_quad_coords, center, r_subdomain)

        W_all.append(W)
        DWx_all.append(DWx)
        DWy_all.append(DWy)

    return W_all, DWx_all, DWy_all, subdomain


def construct_bc_weight_function(
    subdomain_centers, domain_bounds, r_subdomain, subdomain_partition, quad_order
):
    Quad_coord, Quad_weight = gauss_lobatto_jacobi_weights(quad_order)

    # Reshape to (quad_order,) for easier meshgrid/outer product operations
    quad_coord_1d = Quad_coord.squeeze()
    quad_weight_1d = Quad_weight.squeeze()

    bc_W_all, bc_DWx_all, bc_DWy_all = [], [], []
    bc_subdomain_quad_data = []  # To store [coords, weights, jacobians] for each center

    tolerance = 1e-6  # For checking boundary overlap

    for center in subdomain_centers:
        bbox_xmin = jnp.maximum(center[0] - r_subdomain, domain_bounds[0])
        bbox_xmax = jnp.minimum(center[0] + r_subdomain, domain_bounds[1])
        bbox_ymin = jnp.maximum(center[1] - r_subdomain, domain_bounds[2])
        bbox_ymax = jnp.minimum(center[1] + r_subdomain, domain_bounds[3])

        # List to collect boundary quad points for the current center
        current_center_boundary_quad_coords_list = []
        current_center_boundary_quad_weights_list = []  # 1D quad weights
        current_center_boundary_quad_jacobians_list = []  # 1D jacobians

        # Determine actual subdomain lengths after clamping to domain bounds
        subdomain_actual_length_x = bbox_xmax - bbox_xmin
        subdomain_actual_length_y = bbox_ymax - bbox_ymin

        # --- Left Boundary (x = domain_bounds[0]) ---
        if (
            jnp.abs(bbox_xmin - domain_bounds[0]) < tolerance
            and subdomain_actual_length_y > tolerance
        ):
            # Quadrature along Y-direction on the left edge
            bc_segment_ymin = bbox_ymin
            bc_segment_ymax = bbox_ymax
            bc_segment_length_y = bc_segment_ymax - bc_segment_ymin

            # Scale 1D Lobatto points to the y-segment
            scaled_quad_y_coord = bc_segment_ymin + 0.5 * bc_segment_length_y * (
                quad_coord_1d + 1
            )
            scaled_quad_x_coord = jnp.full_like(
                scaled_quad_y_coord, domain_bounds[0]
            )  # x is fixed at domain_bounds[0]

            bc_coords = jnp.stack([scaled_quad_x_coord, scaled_quad_y_coord], axis=-1)
            current_center_boundary_quad_coords_list.append(bc_coords)

            current_center_boundary_quad_weights_list.append(quad_weight_1d)
            current_center_boundary_quad_jacobians_list.append(
                jnp.full_like(quad_weight_1d, 0.5 * bc_segment_length_y)
            )

        # --- Right Boundary (x = domain_bounds[1]) ---
        if (
            jnp.abs(bbox_xmax - domain_bounds[1]) < tolerance
            and subdomain_actual_length_y > tolerance
        ):
            # Quadrature along Y-direction on the right edge
            bc_segment_ymin = bbox_ymin
            bc_segment_ymax = bbox_ymax
            bc_segment_length_y = bc_segment_ymax - bc_segment_ymin

            scaled_quad_y_coord = bc_segment_ymin + 0.5 * bc_segment_length_y * (
                quad_coord_1d + 1
            )
            scaled_quad_x_coord = jnp.full_like(
                scaled_quad_y_coord, domain_bounds[1]
            )  # x is fixed at domain_bounds[1]

            bc_coords = jnp.stack([scaled_quad_x_coord, scaled_quad_y_coord], axis=-1)
            current_center_boundary_quad_coords_list.append(bc_coords)

            current_center_boundary_quad_weights_list.append(quad_weight_1d)
            current_center_boundary_quad_jacobians_list.append(
                jnp.full_like(quad_weight_1d, 0.5 * bc_segment_length_y)
            )

        # --- Bottom Boundary (y = domain_bounds[2]) ---
        if (
            jnp.abs(bbox_ymin - domain_bounds[2]) < tolerance
            and subdomain_actual_length_x > tolerance
        ):
            # Quadrature along X-direction on the bottom edge
            bc_segment_xmin = bbox_xmin
            bc_segment_xmax = bbox_xmax
            bc_segment_length_x = bc_segment_xmax - bc_segment_xmin

            scaled_quad_x_coord = bc_segment_xmin + 0.5 * bc_segment_length_x * (
                quad_coord_1d + 1
            )
            scaled_quad_y_coord = jnp.full_like(
                scaled_quad_x_coord, domain_bounds[2]
            )  # y is fixed at domain_bounds[2]

            bc_coords = jnp.stack([scaled_quad_x_coord, scaled_quad_y_coord], axis=-1)
            current_center_boundary_quad_coords_list.append(bc_coords)

            current_center_boundary_quad_weights_list.append(quad_weight_1d)
            current_center_boundary_quad_jacobians_list.append(
                jnp.full_like(quad_weight_1d, 0.5 * bc_segment_length_x)
            )

        # --- Top Boundary (y = domain_bounds[3]) ---
        if (
            jnp.abs(bbox_ymax - domain_bounds[3]) < tolerance
            and subdomain_actual_length_x > tolerance
        ):
            # Quadrature along X-direction on the top edge
            bc_segment_xmin = bbox_xmin
            bc_segment_xmax = bbox_xmax
            bc_segment_length_x = bc_segment_xmax - bc_segment_xmin

            scaled_quad_x_coord = bc_segment_xmin + 0.5 * bc_segment_length_x * (
                quad_coord_1d + 1
            )
            scaled_quad_y_coord = jnp.full_like(
                scaled_quad_x_coord, domain_bounds[3]
            )  # y is fixed at domain_bounds[3]

            bc_coords = jnp.stack([scaled_quad_x_coord, scaled_quad_y_coord], axis=-1)
            current_center_boundary_quad_coords_list.append(bc_coords)

            current_center_boundary_quad_weights_list.append(quad_weight_1d)
            current_center_boundary_quad_jacobians_list.append(
                jnp.full_like(quad_weight_1d, 0.5 * bc_segment_length_x)
            )

        # Consolidate boundary points for the current center
        if (
            current_center_boundary_quad_coords_list
        ):  # If any boundary points were found for this center
            all_bc_coords_for_center = jnp.concatenate(
                current_center_boundary_quad_coords_list, axis=0
            )
            all_bc_weights_for_center = jnp.concatenate(
                current_center_boundary_quad_weights_list, axis=0
            )
            all_bc_jacobians_for_center = jnp.concatenate(
                current_center_boundary_quad_jacobians_list, axis=0
            )

            # Calculate shape functions from this center to its boundary quad points
            W, DWx, DWy = weight_function(all_bc_coords_for_center, center, r_subdomain)

            bc_W_all.append(W)
            bc_DWx_all.append(DWx)
            bc_DWy_all.append(DWy)
            bc_subdomain_quad_data.append(
                [
                    all_bc_coords_for_center,
                    all_bc_weights_for_center,
                    all_bc_jacobians_for_center,
                ]
            )

        else:
            # If no boundary points, append empty arrays to maintain list length consistency
            bc_W_all.append(jnp.array([]))
            bc_DWx_all.append(jnp.array([]))
            bc_DWy_all.append(jnp.array([]))
            bc_subdomain_quad_data.append(
                [jnp.array([]), jnp.array([]), jnp.array([])]
            )  # Empty data for this center

    return bc_W_all, bc_DWx_all, bc_DWy_all, bc_subdomain_quad_data
