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
from jax import grad
import jaxopt
from flax import struct

import matplotlib.pyplot as plt
from scipy.special import roots_jacobi, jacobi
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import time
import sys
import logging

from quadrature import gauss_lobatto_jacobi_weights
import generate_model
from weight_functions import construct_subdomain_weights, construct_bc_weight_function
from shape_functions import (
    inside_supported_domain,
    get_max_qp_neighbors,
    precompute_shape_functions,
    get_padded_neighbors_and_mask,
    calculate_shape_functions_vmapped_for_subdomain,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

config.update("jax_enable_x64", True)
logger.info("JAX devices: %s", jax.devices())

# ============= model parameters =============#

length_x = 1
length_y = 1
nx_rkpm = 21
ny_rkpm = 21
n_subdomain = 41
subdomain_size_factor = 3.0

domain_bounds = jnp.array([0, 1, 0, 1])

rkpm_nodes, centers, r_subdomain = generate_model.generate_plate(
    domain_bounds,
    nx_rkpm=nx_rkpm,
    ny_rkpm=ny_rkpm,
    n_subdomain=n_subdomain,
    subdomain_size_factor=subdomain_size_factor,
)
num_rkpm_nodes = rkpm_nodes.shape[0]
num_subdomains = centers.shape[0]

logger.info("//-------- Model Parameters--------//")
logger.info("Number of RKPM nodes: %s", num_rkpm_nodes)
logger.info("Number of subdomains: %s", num_subdomains)

# ============= plot Model =============#
plt.figure(figsize=(6, 6))
plt.scatter(rkpm_nodes[:, 0], rkpm_nodes[:, 1])

# ============= material parameters =============#
E = 1000
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
D = jnp.array(
    [
        [lam + 2 * mu, lam, lam, 0.0],
        [lam, lam + 2 * mu, lam, 0.0],
        [lam, lam, lam + 2 * mu, 0.0],
        [0.0, 0.0, 0.0, mu],
    ]
)


# ============= quadrature parameters =============#
quadrature_order = 6
quadrature_points, quadrature_weights = gauss_lobatto_jacobi_weights(quadrature_order)
quadrature_partition = 1

logger.info("Compute weight functions")

W_all, DWx, DWy, subdomain = construct_subdomain_weights(
    centers, domain_bounds, r_subdomain, quadrature_partition, quadrature_order
)

bc_W_all, bc_DWx_all, bc_DWy_all, bc_subdomain_quad_data = construct_bc_weight_function(
    centers, domain_bounds, r_subdomain, quadrature_partition, quadrature_order
)

logger.info("Compute max neighbors")
max_neighbors = get_max_qp_neighbors(subdomain, rkpm_nodes, r_subdomain)

logger.info("Pre-compute shape functions")
phi_all, dphi_all = precompute_shape_functions(
    subdomain, rkpm_nodes, r_subdomain, max_neighbors
)


# ============ define Neural Networks ==============
def MLP(layers, activation=relu):
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)

            lb = -(-1 / jnp.sqrt(d_in))
            ub = 1 / jnp.sqrt(d_out)

            W = lb + (ub - lb) * random.uniform(k1, shape=(d_in, d_out))
            b = random.uniform(k2, shape=(d_out,))

            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = activation(outputs)

        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs

    return init, apply


def gather_disp(nodal_values_padded, ids, mask):
    return nodal_values_padded[ids] * mask[:, None]


def deviator(stress):
    sxx, syy, szz, sxy = stress
    mean_stress = (sxx + syy + szz) / 3.0
    return jnp.array([sxx - mean_stress, syy - mean_stress, szz - mean_stress, sxy])


def voigt_strain(dphi_dx, dphi_dy, disp):
    du_dx = jnp.sum(dphi_dx * disp[:, 0])
    du_dy = jnp.sum(dphi_dy * disp[:, 0])
    dv_dx = jnp.sum(dphi_dx * disp[:, 1])
    dv_dy = jnp.sum(dphi_dy * disp[:, 1])
    return jnp.array([du_dx, dv_dy, 0.0, du_dy + dv_dx])


def j2_return_mapping(
    elastic_trial_strain,
    eqv_p,
    D_material,
    yield_stress,
    hardening_factor,
):
    stress_trial = D_material @ elastic_trial_strain
    s = deviator(stress_trial)
    s_norm = jnp.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + 2.0 * s[3] ** 2)

    safe_s_norm = jnp.where(s_norm > 0.0, s_norm, 1.0)
    normal = s / safe_s_norm
    f = s_norm - jnp.sqrt(2.0 / 3.0) * (yield_stress + hardening_factor * eqv_p)
    plastic_step = f > 0.0

    gamma = jnp.where(
        plastic_step,
        f / (2.0 * mu + 2.0 / 3.0 * hardening_factor),
        0.0,
    )

    stress = stress_trial - 2.0 * mu * gamma * normal

    deps_p = gamma * jnp.array([normal[0], normal[1], normal[2], 2.0 * normal[3]])
    eqv_new = eqv_p + jnp.sqrt(2.0 / 3.0) * gamma

    return stress, eqv_new, deps_p


def von_mises_stress(stress):
    s = deviator(stress)
    s_norm_sq = s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + 2.0 * s[3] ** 2
    return jnp.sqrt(1.5 * s_norm_sq)


# ========== Neural Integrated Meshless (NIM) ==========
class NIM:
    def __init__(self, u_layers, plastic_state):

        self.u_init, self.disp_apply = MLP(u_layers, activation=tanh)
        u_params = self.u_init(random.PRNGKey(1995))
        self.u_params = jax.tree.map(lambda x: x * 0.001, u_params)
        self.input_para = jnp.ones((1, 1))

        self.bc_penalty_factor = 1e5
        self.D_material = D
        self.yield_stress = 50
        self.hardening_factor = 100

        self.rkpm_nodes = rkpm_nodes
        self.max_neighbors = max_neighbors

        self.state = plastic_state

        self.callback_calls = 0

        self.coords_all = jnp.stack([sd[0] for sd in subdomain])
        self.weights_all = jnp.stack([sd[1] for sd in subdomain])
        self.jacobians_all = jnp.stack([sd[2] for sd in subdomain])

        self.dphi_all = jnp.stack(dphi_all)
        self.phi_all = jnp.stack(phi_all)
        self.DWx_all = jnp.stack(DWx)
        self.DWy_all = jnp.stack(DWy)

        self.dphi_dx_all = self.dphi_all[..., 0]
        self.dphi_dy_all = self.dphi_all[..., 1]

        self.traction_subdomains = self.extract_traction(bc_subdomain_quad_data)
        left_roller = self.extract_ebc(
            bc_subdomain_quad_data,
            coord_dim=0,
            bound_index=0,
            constrained_component=0,
            side=0,
        )
        bottom_roller = self.extract_ebc(
            bc_subdomain_quad_data,
            coord_dim=1,
            bound_index=2,
            constrained_component=1,
            side=2,
        )
        self.ebc_subdomains = self.combine_ebc_subdomains(left_roller, bottom_roller)

        self.build_bc_arrays()
        self.precompute_neighbors()

        num_sub, n_ebc = self.ebc_Wb.shape
        self.state = self.state.replace(
            ebc_plastic_strain=jnp.zeros((num_sub, n_ebc, 4)),
            ebc_eqv_plastic_strain=jnp.zeros((num_sub, n_ebc)),
        )

        self.optimizer = jaxopt.ScipyMinimize(
            fun=lambda params, state_pre, current_load: self.loss(
                params, state_pre, current_load
            ),
            method="L-BFGS-B",
            maxiter=10000,
            callback=self.callback,
            jit=True,
            options={
                "maxfun": 30000,
                "maxcor": 30,
                "maxls": 40,
                "ftol": 1e-12,
                "gtol": 1e-12,
            },
        )

        self.build_residual()

    def precompute_neighbors(self):
        def compute_for_subdomain(coords):
            ids, mask = jax.vmap(
                get_padded_neighbors_and_mask, in_axes=(0, None, None, None)
            )(coords, self.rkpm_nodes, r_subdomain, self.max_neighbors)

            return ids, mask

        ids_all, mask_all = jax.vmap(compute_for_subdomain)(self.coords_all)
        ebc_ids_all, ebc_neighbor_mask_all = jax.vmap(compute_for_subdomain)(
            self.ebc_coords
        )

        ebc_phi_all, ebc_dphi_all = jax.vmap(
            lambda coords: calculate_shape_functions_vmapped_for_subdomain(
                coords, self.rkpm_nodes, r_subdomain, self.max_neighbors
            )
        )(self.ebc_coords)

        self.quad_ids_all = ids_all
        self.quad_mask_all = mask_all
        self.ebc_ids_all = ebc_ids_all
        self.ebc_neighbor_mask_all = ebc_neighbor_mask_all
        self.ebc_phi_all = ebc_phi_all
        self.ebc_dphi_all = ebc_dphi_all

    # =========================
    # BC EXTRACTION
    # =========================
    def extract_traction(self, bc_data):
        out = []
        tol = 1e-6
        for i, sub_data in enumerate(bc_data):
            coords = sub_data[0]
            weights = sub_data[1]
            jac = sub_data[2]
            Wb = bc_W_all[i].squeeze()
            if coords.size == 0:
                out.append(None)
                continue

            ids = jnp.where(jnp.isclose(coords[:, 0], domain_bounds[1], atol=tol))[0]

            if ids.shape[0] == 0:
                out.append(None)
            else:
                out.append(
                    {
                        "coords": coords[ids],
                        "weights": weights[ids],
                        "jacobians": jac[ids],
                        "Wb": Wb[ids],
                    }
                )
        return out

    def extract_ebc(self, bc_data, coord_dim, bound_index, constrained_component, side):
        out = []
        tol = 1e-6
        component_mask = jnp.zeros(2).at[constrained_component].set(1.0)
        normal = jnp.zeros(2)
        if side == 0:
            normal = normal.at[0].set(-1.0)
        elif side == 1:
            normal = normal.at[0].set(1.0)
        elif side == 2:
            normal = normal.at[1].set(-1.0)
        elif side == 3:
            normal = normal.at[1].set(1.0)

        for i, sub_data in enumerate(bc_data):
            coords = sub_data[0]
            weights = sub_data[1]
            jac = sub_data[2]
            Wb = bc_W_all[i].squeeze()
            if coords.size == 0:
                out.append(None)
                continue

            ids = jnp.where(
                jnp.isclose(coords[:, coord_dim], domain_bounds[bound_index], atol=tol)
            )[0]

            if ids.shape[0] == 0:
                out.append(None)
            else:
                n = ids.shape[0]
                out.append(
                    {
                        "coords": coords[ids],
                        "weights": weights[ids],
                        "jacobians": jac[ids],
                        "Wb": Wb[ids],
                        "component_mask": jnp.tile(component_mask, (n, 1)),
                        "normal": jnp.tile(normal, (n, 1)),
                    }
                )
        return out

    def combine_ebc_subdomains(self, *ebc_lists):
        out = []
        for entries in zip(*ebc_lists):
            active = [entry for entry in entries if entry is not None]
            if not active:
                out.append(None)
                continue

            out.append(
                {
                    "coords": jnp.concatenate(
                        [entry["coords"] for entry in active], axis=0
                    ),
                    "weights": jnp.concatenate(
                        [entry["weights"] for entry in active], axis=0
                    ),
                    "jacobians": jnp.concatenate(
                        [entry["jacobians"] for entry in active], axis=0
                    ),
                    "Wb": jnp.concatenate([entry["Wb"] for entry in active], axis=0),
                    "component_mask": jnp.concatenate(
                        [entry["component_mask"] for entry in active], axis=0
                    ),
                    "normal": jnp.concatenate(
                        [entry["normal"] for entry in active], axis=0
                    ),
                }
            )
        return out

    def build_bc_arrays(self):
        def pad(data_list, include_component_mask=False):
            max_qp = max([0 if d is None else d["coords"].shape[0] for d in data_list])

            coords, weights, jac, Wb, mask, component_mask, normal = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for d in data_list:
                if d is None:
                    coords.append(jnp.zeros((max_qp, 2)))
                    weights.append(jnp.zeros(max_qp))
                    jac.append(jnp.zeros(max_qp))
                    Wb.append(jnp.zeros(max_qp))
                    mask.append(jnp.zeros(max_qp))
                    if include_component_mask:
                        component_mask.append(jnp.zeros((max_qp, 2)))
                        normal.append(jnp.zeros((max_qp, 2)))
                else:
                    n = d["coords"].shape[0]
                    pad_n = max_qp - n

                    coords.append(jnp.pad(d["coords"], ((0, pad_n), (0, 0))))
                    weights.append(jnp.pad(d["weights"], (0, pad_n)))
                    jac.append(jnp.pad(d["jacobians"], (0, pad_n)))
                    Wb.append(jnp.pad(d["Wb"], (0, pad_n)))
                    mask.append(jnp.concatenate([jnp.ones(n), jnp.zeros(pad_n)]))
                    if include_component_mask:
                        component_mask.append(
                            jnp.pad(d["component_mask"], ((0, pad_n), (0, 0)))
                        )
                        normal.append(jnp.pad(d["normal"], ((0, pad_n), (0, 0))))

            padded = (
                jnp.stack(coords),
                jnp.stack(weights),
                jnp.stack(jac),
                jnp.stack(Wb),
                jnp.stack(mask),
            )

            if include_component_mask:
                return padded + (jnp.stack(component_mask), jnp.stack(normal))
            return padded

        self.tr_coords, self.tr_weights, self.tr_jac, self.tr_Wb, self.tr_mask = pad(
            self.traction_subdomains
        )

        (
            self.ebc_coords,
            self.ebc_weights,
            self.ebc_jac,
            self.ebc_Wb,
            self.ebc_mask,
            self.ebc_component_mask,
            self.ebc_normal,
        ) = pad(self.ebc_subdomains, include_component_mask=True)

    def build_residual(self):

        disp_apply = self.disp_apply
        input_para = self.input_para

        bc_penalty_factor = self.bc_penalty_factor
        D_material = self.D_material
        rkpm_nodes = self.rkpm_nodes
        max_neighbors = self.max_neighbors
        yield_stress = self.yield_stress
        hardening_factor = self.hardening_factor

        @jax.jit
        def residual_fn(
            params,
            state_pre,
            current_load,
            coords_all,
            weights_all,
            jacobians_all,
            dphi_all_x,
            dphi_all_y,
            DWx_all,
            DWy_all,
            tr_Wb,
            tr_weights,
            tr_jac,
            tr_mask,
            ebc_coords,
            ebc_Wb,
            ebc_weights,
            ebc_jac,
            ebc_mask,
            ebc_component_mask,
            ebc_normal,
            quad_ids_all,
            quad_mask_all,
            ebc_ids_all,
            ebc_neighbor_mask_all,
            ebc_phi_all,
            ebc_dphi_all,
        ):
            plastic_strain_all = state_pre.plastic_strain
            eqv_all = state_pre.eqv_plastic_strain
            ebc_plastic_all = state_pre.ebc_plastic_strain
            ebc_eqv_all = state_pre.ebc_eqv_plastic_strain

            nodal_values = disp_apply(params, input_para).reshape(-1, 2)
            nodal_values_padded = jnp.vstack([nodal_values, jnp.zeros((1, 2))])

            gather = partial(gather_disp, nodal_values_padded)

            def qp_kernel(dphi_dx, dphi_dy, disp, dwdx, dwdy, w, J, eps_p, eqv_p):
                elastic_trial_strain = voigt_strain(dphi_dx, dphi_dy, disp) - eps_p

                stress, _, _ = j2_return_mapping(
                    elastic_trial_strain,
                    eqv_p,
                    D_material,
                    yield_stress,
                    hardening_factor,
                )

                r = jnp.array(
                    [
                        dwdx * stress[0] + dwdy * stress[3],
                        dwdx * stress[3] + dwdy * stress[1],
                    ]
                )

                return r * w * J

            def ebc_weak(
                w, dphi_dx, dphi_dy, disp, weight, J, eps_p, eqv_p, constrained_normal
            ):
                elastic_trial_strain = voigt_strain(dphi_dx, dphi_dy, disp) - eps_p
                stress, _, _ = j2_return_mapping(
                    elastic_trial_strain,
                    eqv_p,
                    D_material,
                    yield_stress,
                    hardening_factor,
                )

                nx, ny = constrained_normal

                traction = jnp.array(
                    [
                        stress[0] * nx + stress[3] * ny,
                        stress[3] * nx + stress[1] * ny,
                    ]
                )

                return w * traction * weight * J

            def subdomain_fn(
                coords,
                weights,
                jacobians,
                dphi_x,
                dphi_y,
                DWx_i,
                DWy_i,
                tr_Wb,
                tr_w,
                tr_j,
                tr_mask,
                ebc_coords,
                ebc_Wb,
                ebc_w,
                ebc_j,
                ebc_mask,
                ebc_component_mask,
                ebc_normal,
                quad_ids_i,
                quad_mask_i,
                ebc_ids_i,
                ebc_neighbor_mask_i,
                phi_b,
                dphi_b,
                eps_p_sub,
                eqv_sub,
                ebc_eps_p_sub,
                ebc_eqv_sub,
            ):

                disp_neighbors = jax.vmap(gather)(quad_ids_i, quad_mask_i)

                R_internal = jax.vmap(qp_kernel)(
                    dphi_x,
                    dphi_y,
                    disp_neighbors,
                    DWx_i.squeeze(),
                    DWy_i.squeeze(),
                    weights,
                    jacobians,
                    eps_p_sub,
                    eqv_sub,
                )

                R_internal_sum = jnp.sum(R_internal, axis=0)

                # traction
                traction_vec = jnp.array([current_load, 0.0])
                R_tr = jax.vmap(lambda w, wt, J, m: m * (w * traction_vec * wt * J))(
                    tr_Wb, tr_w, tr_j, tr_mask
                )

                R_traction = jnp.sum(R_tr, axis=0)

                # EBC
                disp_neighbors_b = jax.vmap(gather)(ebc_ids_i, ebc_neighbor_mask_i)

                u_qp = jax.vmap(lambda phi, disp: jnp.sum(phi[:, None] * disp, axis=0))(
                    phi_b, disp_neighbors_b
                )

                R_ebc_weak_qp = jax.vmap(ebc_weak)(
                    ebc_Wb,
                    dphi_b[:, :, 0],
                    dphi_b[:, :, 1],
                    disp_neighbors_b,
                    ebc_w,
                    ebc_j,
                    ebc_eps_p_sub,
                    ebc_eqv_sub,
                    ebc_normal,
                )
                R_ebc_weak = jnp.sum(R_ebc_weak_qp, axis=0)

                u_bar = jnp.array([0.0, 0.0])

                R_ebc_qp = jax.vmap(
                    lambda w, u, wt, J, m, c: m * (w * c * (u - u_bar) * wt * J)
                )(ebc_Wb, u_qp, ebc_w, ebc_j, ebc_mask, ebc_component_mask)

                R_ebc = jnp.sum(R_ebc_qp, axis=0)

                # Pure penalty enforcement of u = 0 on the left boundary.
                # With the residual convention external - internal, the
                # penalty reaction must oppose nonzero boundary displacement.
                R_s = (
                    R_traction - R_internal_sum - bc_penalty_factor * R_ebc - R_ebc_weak
                )

                return jnp.sum(R_s**2)

            loss_all = jax.vmap(subdomain_fn)(
                coords_all,
                weights_all,
                jacobians_all,
                dphi_all_x,
                dphi_all_y,
                DWx_all,
                DWy_all,
                tr_Wb,
                tr_weights,
                tr_jac,
                tr_mask,
                ebc_coords,
                ebc_Wb,
                ebc_weights,
                ebc_jac,
                ebc_mask,
                ebc_component_mask,
                ebc_normal,
                quad_ids_all,
                quad_mask_all,
                ebc_ids_all,
                ebc_neighbor_mask_all,
                ebc_phi_all,
                ebc_dphi_all,
                plastic_strain_all,
                eqv_all,
                ebc_plastic_all,
                ebc_eqv_all,
            )

            return jnp.mean(loss_all)

        self.residual_fn = residual_fn

    def get_static_data(self):
        return (
            self.coords_all,
            self.weights_all,
            self.jacobians_all,
            self.dphi_dx_all,
            self.dphi_dy_all,
            self.DWx_all,
            self.DWy_all,
            self.tr_Wb,
            self.tr_weights,
            self.tr_jac,
            self.tr_mask,
            self.ebc_coords,
            self.ebc_Wb,
            self.ebc_weights,
            self.ebc_jac,
            self.ebc_mask,
            self.ebc_component_mask,
            self.ebc_normal,
            self.quad_ids_all,
            self.quad_mask_all,
            self.ebc_ids_all,
            self.ebc_neighbor_mask_all,
            self.ebc_phi_all,
            self.ebc_dphi_all,
        )

    def loss(self, params, state_pre, current_load):
        return self.residual_fn(
            params, state_pre, current_load, *self.get_static_data()
        )

    def update_plastic_state(self, params, state_pre):
        D_material = self.D_material
        yield_stress = self.yield_stress
        hardening_factor = self.hardening_factor

        nodal_values = self.disp_apply(params, self.input_para).reshape(-1, 2)
        nodal_values_padded = jnp.vstack([nodal_values, jnp.zeros((1, 2))])

        gather = partial(gather_disp, nodal_values_padded)

        def return_map_qp(dphi_dx, dphi_dy, disp, eps_p, eqv_p):
            elastic_trial_strain = voigt_strain(dphi_dx, dphi_dy, disp) - eps_p
            stress_new, eqv_new, deps_p = j2_return_mapping(
                elastic_trial_strain,
                eqv_p,
                D_material,
                yield_stress,
                hardening_factor,
            )
            eps_p_new = eps_p + deps_p
            return eps_p_new, eqv_new, stress_new

        def subdomain_update(
            dphi_x, dphi_y, quad_ids_i, quad_mask_i, eps_p_sub, eqv_sub
        ):
            disp_neighbors = jax.vmap(gather)(quad_ids_i, quad_mask_i)
            return jax.vmap(return_map_qp)(
                dphi_x, dphi_y, disp_neighbors, eps_p_sub, eqv_sub
            )

        eps_p_new, eqv_new, stress_new = jax.vmap(subdomain_update)(
            self.dphi_dx_all,
            self.dphi_dy_all,
            self.quad_ids_all,
            self.quad_mask_all,
            state_pre.plastic_strain,
            state_pre.eqv_plastic_strain,
        )

        # same return map at the EBC quadrature points, using the precomputed
        # EBC shape-function derivatives / neighbours
        ebc_eps_p_new, ebc_eqv_new, _ = jax.vmap(subdomain_update)(
            self.ebc_dphi_all[..., 0],
            self.ebc_dphi_all[..., 1],
            self.ebc_ids_all,
            self.ebc_neighbor_mask_all,
            state_pre.ebc_plastic_strain,
            state_pre.ebc_eqv_plastic_strain,
        )

        return PlasticVariables(
            plastic_strain=eps_p_new,
            eqv_plastic_strain=eqv_new,
            stress=stress_new,
            ebc_plastic_strain=ebc_eps_p_new,
            ebc_eqv_plastic_strain=ebc_eqv_new,
        )

    def train_full_flow(self, load_step, state=None, final_load=10.0):
        # default to the EBC-augmented state built in __init__
        if state is None:
            state = self.state
        for i in range(load_step):
            load_scale = (i + 1) / load_step
            logger.info(f"Load step: {i + 1}, load scale: {load_scale}")
            current_load = load_scale * final_load

            state = self.train_single_step(state, current_load)
        self.state = state
        return state

    def train_single_step(self, state_old, current_load):
        logger.info("Starting NIM training...")
        sys.stdout.flush()
        self.i_opt = 0
        self.start_time = time.time()

        self.state_pre = state_old
        self.current_load = current_load

        try:
            sol = self.optimizer.run(self.u_params, state_old, current_load)
            self.u_params = sol.params  # Update params after optimization
            self.solution = sol
        except Exception as e:
            logger.exception(f"An error occurred during NIM training: {e}")
            raise

        new_state = self.update_plastic_state(self.u_params, state_old)
        eqv_increment = jnp.max(
            jnp.abs(new_state.eqv_plastic_strain - state_old.eqv_plastic_strain)
        )
        ebc_eqv_increment = jnp.max(
            jnp.abs(new_state.ebc_eqv_plastic_strain - state_old.ebc_eqv_plastic_strain)
        )
        logger.info(new_state.eqv_plastic_strain[200])
        logger.info(state_old.eqv_plastic_strain[200])
        logger.info(
            "old vm stress[200]: %s", jax.vmap(von_mises_stress)(state_old.stress[200])
        )
        logger.info(
            "new vm stress[200]: %s", jax.vmap(von_mises_stress)(new_state.stress[200])
        )
        logger.info(new_state.stress[200])
        plastic_increment = float(jnp.maximum(eqv_increment, ebc_eqv_increment))

        logger.info("Plastic state increment: %.6e", plastic_increment)
        self.state_pre = new_state
        self.state = new_state
        logger.info("NIM training completed successfully.")
        return new_state

    def callback(self, params):
        # Callback to print progress during optimization
        self.i_opt += 1
        self.callback_calls += 1  # Increment the counter
        # Use state.value to get the concrete loss value
        if self.i_opt % 1000 == 0:
            loss_val = self.loss(params, self.state_pre, self.current_load)
            logger.info(
                f"Iteration {self.i_opt}, Loss: {loss_val:.6e}, Time: {time.time() - self.start_time:.2f}s"
            )
        return

    def predict(self, params, coords, return_von_mises=False):
        coords = jnp.asarray(coords)
        single_point = coords.ndim == 1
        coords_eval = jnp.atleast_2d(coords)

        phi, dphi = calculate_shape_functions_vmapped_for_subdomain(
            coords_eval, self.rkpm_nodes, r_subdomain, self.max_neighbors
        )

        ids, mask = jax.vmap(
            get_padded_neighbors_and_mask, in_axes=(0, None, None, None)
        )(coords_eval, self.rkpm_nodes, r_subdomain, self.max_neighbors)

        nodal = self.disp_apply(params, self.input_para).reshape(-1, 2)
        nodal = jnp.vstack([nodal, jnp.zeros((1, 2))])

        neighbor_disp = jax.vmap(lambda i, m: nodal[i] * m[:, None])(ids, mask)
        disp = jnp.sum(phi[:, :, None] * neighbor_disp, axis=1)

        if not return_von_mises:
            if single_point:
                return disp[0]
            return disp

        strain = jax.vmap(voigt_strain)(
            dphi[:, :, 0],
            dphi[:, :, 1],
            neighbor_disp,
        )
        stress = jax.vmap(lambda eps: self.D_material @ eps)(strain)
        vm = jax.vmap(von_mises_stress)(stress)

        if single_point:
            return disp[0], vm[0]
        return disp, vm

    def compute_von_mises(self, state=None):
        if state is None:
            state = self.state
        return jax.vmap(jax.vmap(von_mises_stress))(state.stress)


def average_duplicate_points(coords, values, decimals=12):
    rounded_coords = np.round(coords, decimals=decimals)
    unique_coords, inverse = np.unique(rounded_coords, axis=0, return_inverse=True)
    value_sum = np.zeros(unique_coords.shape[0], dtype=float)
    value_count = np.zeros(unique_coords.shape[0], dtype=float)

    np.add.at(value_sum, inverse, values)
    np.add.at(value_count, inverse, 1.0)

    return unique_coords, value_sum / value_count


def committed_von_mises_on_grid(
    model,
    X,
    Y,
    state=None,
    interpolation="linear",
    smooth_sigma=1.5,
):
    if state is None:
        state = model.state

    coords = np.asarray(jax.device_get(model.coords_all)).reshape(-1, 2)
    vm_qp = np.asarray(jax.device_get(model.compute_von_mises(state))).reshape(-1)

    coords, vm_qp = average_duplicate_points(coords, vm_qp)
    vm_grid = griddata(
        coords, vm_qp, (np.asarray(X), np.asarray(Y)), method=interpolation
    )

    # Fill any tiny interpolation holes near boundaries with nearest-neighbor values.
    missing = np.isnan(vm_grid)
    if np.any(missing):
        vm_nearest = griddata(
            coords, vm_qp, (np.asarray(X), np.asarray(Y)), method="nearest"
        )
        vm_grid[missing] = vm_nearest[missing]

    if smooth_sigma is not None and smooth_sigma > 0.0:
        vm_grid = gaussian_filter(vm_grid, sigma=smooth_sigma)

    return vm_grid


def print_plastic_state_summary(model, state=None):
    if state is None:
        state = model.state

    eqp = state.eqv_plastic_strain.reshape(-1)
    vm = model.compute_von_mises(state).reshape(-1)
    current_yield = model.yield_stress + model.hardening_factor * eqp
    yielded = eqp > 1e-10

    logger.info("Plastic state summary:")
    logger.info("  max eqp: %.6e", float(jnp.max(eqp)))
    logger.info("  mean eqp: %.6e", float(jnp.mean(eqp)))
    logger.info("  yielded fraction from eqp: %.2f%%", float(100.0 * jnp.mean(yielded)))
    logger.info("  max committed von Mises: %.6e", float(jnp.max(vm)))
    logger.info("  max current yield stress: %.6e", float(jnp.max(current_yield)))


def print_von_mises_uniformity_summary(model, state=None, boundary_margin=0.08):
    if state is None:
        state = model.state

    coords = model.coords_all.reshape(-1, 2)
    vm = model.compute_von_mises(state).reshape(-1)
    interior = (
        (coords[:, 0] > domain_bounds[0] + boundary_margin)
        & (coords[:, 0] < domain_bounds[1] - boundary_margin)
        & (coords[:, 1] > domain_bounds[2] + boundary_margin)
        & (coords[:, 1] < domain_bounds[3] - boundary_margin)
    )
    vm_i = vm[interior]
    mean_vm = jnp.mean(vm_i)
    cov_vm = jnp.std(vm_i) / (mean_vm + 1e-14)

    logger.info("Von Mises uniformity summary:")
    logger.info("  interior mean vm: %.6e", float(mean_vm))
    logger.info("  interior min vm: %.6e", float(jnp.min(vm_i)))
    logger.info("  interior max vm: %.6e", float(jnp.max(vm_i)))
    logger.info("  interior coeff. of variation: %.6e", float(cov_vm))


def print_boundary_displacement_summary(model, n_points=101):
    y = jnp.linspace(domain_bounds[2], domain_bounds[3], n_points)
    x = jnp.linspace(domain_bounds[0], domain_bounds[1], n_points)
    left_points = jnp.stack([jnp.full_like(y, domain_bounds[0]), y], axis=-1)
    bottom_points = jnp.stack([x, jnp.full_like(x, domain_bounds[2])], axis=-1)
    right_points = jnp.stack([jnp.full_like(y, domain_bounds[1]), y], axis=-1)

    u_left = model.predict(model.u_params, left_points)
    u_bottom = model.predict(model.u_params, bottom_points)
    u_right = model.predict(model.u_params, right_points)

    logger.info("Boundary displacement summary:")
    logger.info(
        "  max |ux| on left roller: %.6e", float(jnp.max(jnp.abs(u_left[:, 0])))
    )
    logger.info(
        "  max |uy| on bottom roller: %.6e",
        float(jnp.max(jnp.abs(u_bottom[:, 1]))),
    )
    logger.info("  mean ux on right edge: %.6e", float(jnp.mean(u_right[:, 0])))
    logger.info(
        "  max |uy| on right edge: %.6e", float(jnp.max(jnp.abs(u_right[:, 1])))
    )


@struct.dataclass
class PlasticVariables:
    plastic_strain: jnp.ndarray
    eqv_plastic_strain: jnp.ndarray
    stress: jnp.ndarray
    ebc_plastic_strain: jnp.ndarray = None
    ebc_eqv_plastic_strain: jnp.ndarray = None


state = PlasticVariables(
    plastic_strain=jnp.stack([jnp.zeros((sd[0].shape[0], 4)) for sd in subdomain]),
    eqv_plastic_strain=jnp.stack([jnp.zeros((sd[0].shape[0],)) for sd in subdomain]),
    stress=jnp.stack([jnp.zeros((sd[0].shape[0], 4)) for sd in subdomain]),
)

load_step = 20
u_layers = [1, 10, 2 * num_rkpm_nodes]
nim_model = NIM(u_layers, plastic_state=state)
nim_model.train_full_flow(load_step=load_step)

x_test = jnp.linspace(0, length_x, 21)
y_test = jnp.linspace(0, length_y, 21)

X_pre, Y_pre = jnp.meshgrid(x_test, y_test)

test_points = jnp.stack((X_pre.ravel(), Y_pre.ravel()), axis=-1)

u_pred = nim_model.predict(nim_model.u_params, test_points)

ux = u_pred[:, 0].reshape(X_pre.shape)
uy = u_pred[:, 1].reshape(Y_pre.shape)
vm = committed_von_mises_on_grid(
    nim_model,
    X_pre,
    Y_pre,
    interpolation="linear",
    smooth_sigma=0.75,
)
print_plastic_state_summary(nim_model)
print_von_mises_uniformity_summary(nim_model)
print_boundary_displacement_summary(nim_model)

import matplotlib.pyplot as plt

plt.figure()
plt.contourf(X_pre, Y_pre, ux, levels=100, cmap="jet")
plt.colorbar()
plt.title("Displacement $u_x$")
plt.xlabel("x")
plt.ylabel("y")

plt.figure()
plt.contourf(X_pre, Y_pre, uy, levels=100, cmap="jet")
plt.colorbar()
plt.title("Displacement $u_y$")
plt.xlabel("x")
plt.ylabel("y")

plt.figure()
plt.contourf(X_pre, Y_pre, vm, levels=100, cmap="jet")
plt.colorbar()
plt.title("Von Mises stress")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
