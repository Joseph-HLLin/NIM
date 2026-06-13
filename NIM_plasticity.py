import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as jnp

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
subdomain_size_factor = 2.5

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
D = (E / (1 - nu**2)) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


# ============= quadrature parameters =============#
quadrature_order = 5
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


def deviator(t):
    return t - jnp.trace(t) / 2 * jnp.eye(2)


def voigt_strain(dphi_dx, dphi_dy, disp):
    du_dx = jnp.sum(dphi_dx * disp[:, 0])
    du_dy = jnp.sum(dphi_dy * disp[:, 0])
    dv_dx = jnp.sum(dphi_dx * disp[:, 1])
    dv_dy = jnp.sum(dphi_dy * disp[:, 1])
    return jnp.array([du_dx, dv_dy, du_dy + dv_dx])


def j2_return_mapping(strain, eqv_p, D_material, yield_stress, hardening_factor):
    stress_trial = D_material @ strain
    stress_trial_m = jnp.array(
        [[stress_trial[0], stress_trial[2]], [stress_trial[2], stress_trial[1]]]
    )
    s = deviator(stress_trial_m)
    s_norm = jnp.linalg.norm(s)
    normal = s / (s_norm + 1e-8)

    f = s_norm - jnp.sqrt(2 / 3) * (yield_stress + hardening_factor * eqv_p)
    gamma = jnp.maximum(f, 0.0) / (2 * mu + 2 / 3 * hardening_factor)

    stress_m = stress_trial_m - 2 * mu * gamma * normal
    stress = jnp.array([stress_m[0, 0], stress_m[1, 1], stress_m[0, 1]])

    deps_p = gamma * jnp.array([normal[0, 0], normal[1, 1], 2.0 * normal[0, 1]])
    eqv_new = eqv_p + jnp.sqrt(2 / 3) * gamma

    return stress, eqv_new, deps_p


# ========== Neural Integraed Meshless (NIM) ==========
class NIM:
    def __init__(self, u_layers, plastic_state):

        self.u_init, self.disp_apply = MLP(u_layers, activation=tanh)
        u_params = self.u_init(random.PRNGKey(1995))
        self.u_params = jax.tree.map(lambda x: x * 0.001, u_params)
        self.input_para = jnp.ones((1, 1))

        self.bc_penalty_factor = 1e5
        self.D_material = D
        self.yield_stress = 8
        self.hardening_factor = 1

        self.rkpm_nodes = rkpm_nodes
        self.max_neighbors = max_neighbors

        self.state = plastic_state

        self.callback_calls = 0

        self.coords_all = jnp.stack([sd[0] for sd in subdomain])
        self.weights_all = jnp.stack([sd[1] for sd in subdomain])
        self.jacobians_all = jnp.stack([sd[2] for sd in subdomain])

        self.dphi_all = jnp.stack(dphi_all)
        self.DWx_all = jnp.stack(DWx)
        self.DWy_all = jnp.stack(DWy)

        self.dphi_dx_all = self.dphi_all[..., 0]
        self.dphi_dy_all = self.dphi_all[..., 1]

        self.traction_subdomains = self.extract_traction(bc_subdomain_quad_data)
        self.ebc_subdomains = self.extract_ebc(bc_subdomain_quad_data)

        self.build_bc_arrays()
        self.precompute_neighbors()

        # The passed-in plastic_state only carries the interior-quadrature
        # history. The EBC weak term lives on the padded EBC quadrature points
        # (self.ebc_Wb has shape (num_subdomains, n_ebc_padded)), so allocate a
        # matching zero plastic history there and fold it into the state.
        num_sub, n_ebc = self.ebc_Wb.shape
        self.state = self.state.replace(
            ebc_plastic_strain=jnp.zeros((num_sub, n_ebc, 3)),
            ebc_eqv_plastic_strain=jnp.zeros((num_sub, n_ebc)),
        )

        self.optimizer = jaxopt.ScipyMinimize(
            fun=lambda params, state_pre, current_load: self.loss(
                params, state_pre, current_load
            ),
            method="L-BFGS-B",
            maxiter=100000,
            callback=self.callback,
            jit=True,
            options={
                "maxfun": 100000,
                "maxcor": 100,
                "maxls": 100,
                "ftol": 1e-15,
                "gtol": 1e-15,
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

    def extract_ebc(self, bc_data):
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

            ids = jnp.where(jnp.isclose(coords[:, 0], domain_bounds[0], atol=tol))[0]

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

    def build_bc_arrays(self):
        def pad(data_list):
            max_qp = max([0 if d is None else d["coords"].shape[0] for d in data_list])

            coords, weights, jac, Wb, mask = [], [], [], [], []

            for d in data_list:
                if d is None:
                    coords.append(jnp.zeros((max_qp, 2)))
                    weights.append(jnp.zeros(max_qp))
                    jac.append(jnp.zeros(max_qp))
                    Wb.append(jnp.zeros(max_qp))
                    mask.append(jnp.zeros(max_qp))
                else:
                    n = d["coords"].shape[0]
                    pad_n = max_qp - n

                    coords.append(jnp.pad(d["coords"], ((0, pad_n), (0, 0))))
                    weights.append(jnp.pad(d["weights"], (0, pad_n)))
                    jac.append(jnp.pad(d["jacobians"], (0, pad_n)))
                    Wb.append(jnp.pad(d["Wb"], (0, pad_n)))
                    mask.append(jnp.concatenate([jnp.ones(n), jnp.zeros(pad_n)]))

            return (
                jnp.stack(coords),
                jnp.stack(weights),
                jnp.stack(jac),
                jnp.stack(Wb),
                jnp.stack(mask),
            )

        self.tr_coords, self.tr_weights, self.tr_jac, self.tr_Wb, self.tr_mask = pad(
            self.traction_subdomains
        )

        (
            self.ebc_coords,
            self.ebc_weights,
            self.ebc_jac,
            self.ebc_Wb,
            self.ebc_mask,
        ) = pad(self.ebc_subdomains)

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
                strain = voigt_strain(dphi_dx, dphi_dy, disp) - eps_p

                stress, _, _ = j2_return_mapping(
                    strain, eqv_p, D_material, yield_stress, hardening_factor
                )

                r = jnp.array(
                    [
                        dwdx * stress[0] + dwdy * stress[2],
                        dwdx * stress[2] + dwdy * stress[1],
                    ]
                )

                return r * w * J

            def ebc_weak(w, dphi_dx, dphi_dy, disp, weight, J, eps_p, eqv_p):
                strain = voigt_strain(dphi_dx, dphi_dy, disp) - eps_p
                stress, _, _ = j2_return_mapping(
                    strain, eqv_p, D_material, yield_stress, hardening_factor
                )

                nx, ny = -1.0, 0.0

                traction = jnp.array(
                    [
                        stress[0] * nx + stress[2] * ny,
                        stress[2] * nx + stress[1] * ny,
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

                u_bar = jnp.array([0.0, 0.0])

                R_ebc_qp = jax.vmap(
                    lambda w, u, wt, J, m: m * (w * (u - u_bar) * wt * J)
                )(ebc_Wb, u_qp, ebc_w, ebc_j, ebc_mask)

                R_ebc = jnp.sum(R_ebc_qp, axis=0)

                # plastic history tracked on the EBC quadrature points
                R_ebc_weak_qp = jax.vmap(ebc_weak)(
                    ebc_Wb,
                    dphi_b[:, :, 0],
                    dphi_b[:, :, 1],
                    disp_neighbors_b,
                    ebc_w,
                    ebc_j,
                    ebc_eps_p_sub,
                    ebc_eqv_sub,
                )
                R_ebc_weak = jnp.sum(R_ebc_weak_qp, axis=0)

                R_s = (
                    R_traction - R_internal_sum + bc_penalty_factor * R_ebc + R_ebc_weak
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
            self.quad_ids_all,
            self.quad_mask_all,
            self.ebc_ids_all,
            self.ebc_neighbor_mask_all,
            self.ebc_phi_all,
            self.ebc_dphi_all,
        )

    def loss(self, params, state_pre, current_load):
        # state_pre and current_load are passed in (not read from self) so the
        # optimizer's jitted objective treats them as traced inputs that change
        # each load step, instead of baking in the first step's values.
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
            strain = voigt_strain(dphi_dx, dphi_dy, disp) - eps_p
            stress_new, eqv_new, deps_p = j2_return_mapping(
                strain, eqv_p, D_material, yield_stress, hardening_factor
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

    def train_full_flow(self, load_step, state=None):
        # default to the EBC-augmented state built in __init__
        if state is None:
            state = self.state
        for i in range(load_step):
            load_scale = (i + 1) / load_step
            logger.info(f"Load step: {i + 1}, load scale: {load_scale}")
            self.current_load = load_scale * 10
            self.state_pre = state

            state = self.train_single_step()  # carry the updated state forward
        self.state = state
        return state

    def train_single_step(self):
        logger.info("Starting NIM training...")
        sys.stdout.flush()
        self.i_opt = 0
        self.start_time = time.time()
        try:
            sol = self.optimizer.run(self.u_params, self.state_pre, self.current_load)
            self.u_params = sol.params  # Update params after optimization
            self.solution = sol
            logger.info("NIM training completed successfully.")
        except Exception as e:
            logger.info(f"An error occurred during NIM training: {e}")
        # write side of the plastic update, from the converged displacement
        new_state = self.update_plastic_state(self.u_params, self.state_pre)
        return new_state

    def callback(self, params):
        # Callback to print progress during optimization
        self.i_opt += 1
        self.callback_calls += 1  # Increment the counter
        # Use state.value to get the concrete loss value
        if self.i_opt % 100 == 0:
            loss_val = self.loss(params, self.state_pre, self.current_load)
            logger.info(
                f"Iteration {self.i_opt}, Loss: {loss_val:.6e}, Time: {time.time() - self.start_time:.2f}s"
            )
        return

    def predict(self, params, coords):
        phi, _ = calculate_shape_functions_vmapped_for_subdomain(
            coords, rkpm_nodes, r_subdomain, max_neighbors
        )

        ids, mask = jax.vmap(
            get_padded_neighbors_and_mask, in_axes=(0, None, None, None)
        )(coords, rkpm_nodes, r_subdomain, max_neighbors)

        nodal = self.disp_apply(params, self.input_para).reshape(-1, 2)
        nodal = jnp.vstack([nodal, jnp.zeros((1, 2))])

        disp = jax.vmap(lambda i, m: nodal[i] * m[:, None])(ids, mask)

        return jnp.sum(phi[:, :, None] * disp, axis=1)


@struct.dataclass
class PlasticVariables:
    plastic_strain: jnp.ndarray
    eqv_plastic_strain: jnp.ndarray
    stress: jnp.ndarray
    # plastic history on the padded EBC quadrature points (filled in by NIM,
    # once the padded EBC shape is known)
    ebc_plastic_strain: jnp.ndarray = None
    ebc_eqv_plastic_strain: jnp.ndarray = None


state = PlasticVariables(
    plastic_strain=jnp.stack([jnp.zeros((sd[0].shape[0], 3)) for sd in subdomain]),
    eqv_plastic_strain=jnp.stack([jnp.zeros((sd[0].shape[0],)) for sd in subdomain]),
    stress=jnp.stack([jnp.zeros((sd[0].shape[0], 3)) for sd in subdomain]),
)

load_step = 1
u_layers = [1, 10, 2 * num_rkpm_nodes]
nim_model = NIM(u_layers, plastic_state=state)
nim_model.train_full_flow(load_step=load_step)

# After training, you can check the counter
logger.info(f"Callback was called {nim_model.callback_calls} times during training.")

x_test = jnp.linspace(0, length_x, 501)
y_test = jnp.linspace(0, length_y, 501)

X_pre, Y_pre = jnp.meshgrid(x_test, y_test)

test_points = jnp.stack((X_pre.ravel(), Y_pre.ravel()), axis=-1)

u_pred = nim_model.predict(nim_model.u_params, test_points)

ux = u_pred[:, 0].reshape(X_pre.shape)
uy = u_pred[:, 1].reshape(Y_pre.shape)

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
plt.show()
