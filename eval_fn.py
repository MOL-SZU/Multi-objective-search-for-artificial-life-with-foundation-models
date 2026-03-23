import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from rollout import rollout_simulation


def encode_prompts(fm, prompts: list[str]) -> list[jnp.ndarray]:
    """
    Pre-compute text embeddings for all prompts.
    If fm.encode_text already returns L2-normalised vectors,
    the division is a no-op but causes no harm.
    Returns a list of jnp.ndarray, each shape (1, D).
    """
    z_txt_list = []
    for p in prompts:
        z = fm.embed_txt(p)
        z = jnp.atleast_2d(z)
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        z_txt_list.append(z)
    return z_txt_list


def build_rollout_fn(substrate, fm, time_sampling=8):
    """
    Build and jit the rollout function.
    time_sampling is exposed as a parameter rather than hard-coded.

    Parameters
    ----------
    substrate     : ASAL substrate instance
    fm            : foundation model instance
    time_sampling : number of frames sampled evenly across the trajectory
    """
    inner_sub     = getattr(substrate, 'substrate', substrate)
    rollout_steps = getattr(inner_sub, 'rollout_steps', 256)
    img_size      = getattr(inner_sub, 'grid_size', 224)

    raw_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=rollout_steps,
        time_sampling=(time_sampling, True),
        img_size=img_size,
        return_state=False,
    )
    return jax.jit(raw_fn)


def get_substrate_bounds(param_cls) -> tuple[np.ndarray, np.ndarray]:
    """
    FlattenSubstrateParameters does not provide explicit bounds.
    Lenia parameters range approximately [-12, 8], far outside [0, 1].
    Return None so compute_bounds_from_seeds uses only seed-based soft bounds.
    """
    return None, None


def compute_bounds_from_seeds(
    substrate,
    seeds: np.ndarray,
    dyn_margin: float = 0.5,
    init_margin: float = 1.0,
    global_xl: np.ndarray | None = None,
    global_xu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension search bounds anchored at single-objective optima.
    The resulting soft bounds are intersected with the substrate hard bounds
    to avoid invalid parameter regions.

    Parameters
    ----------
    substrate   : ASAL substrate instance
    seeds       : (N, n_params) array of single-objective optimal solutions
    dyn_margin  : expansion margin for dynamic-rule parameter dimensions
    init_margin : expansion margin for initial-state parameter dimensions
    global_xl   : hard lower bounds from get_substrate_bounds() (optional)
    global_xu   : hard upper bounds from get_substrate_bounds() (optional)
    """
    seeds     = np.atleast_2d(seeds).astype(np.float64)
    inner_sub = getattr(substrate, 'substrate', substrate)
    n_var     = seeds.shape[1]

    # If substrate has no n_params_dyn, treat all dimensions as
    # dynamic-rule parameters (e.g. Lenia has no separate init-state params)
    DYN_DIM = int(getattr(inner_sub, 'n_params_dyn', n_var))

    xl = np.empty(n_var, dtype=np.float64)
    xu = np.empty(n_var, dtype=np.float64)

    # Dynamic-rule dimensions
    xl[:DYN_DIM] = seeds[:, :DYN_DIM].min(axis=0) - dyn_margin
    xu[:DYN_DIM] = seeds[:, :DYN_DIM].max(axis=0) + dyn_margin

    # Initial-state dimensions (only if substrate distinguishes them)
    if DYN_DIM < n_var:
        xl[DYN_DIM:] = seeds[:, DYN_DIM:].min(axis=0) - init_margin
        xu[DYN_DIM:] = seeds[:, DYN_DIM:].max(axis=0) + init_margin
        print(f"[Bounds] dyn_dim={DYN_DIM}, init_dim={n_var - DYN_DIM}")
    else:
        print(f"[Bounds] All {n_var} dimensions treated as dynamic-rule parameters.")

    if global_xl is not None:
        xl = np.maximum(xl, global_xl)
    if global_xu is not None:
        xu = np.minimum(xu, global_xu)

    return xl, xu

def generate_parents_from_optimum(
    x_star: np.ndarray,
    n_parents: int,
    noise_scale: float = 0.05,
    xl: np.ndarray | None = None,
    xu: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate parent candidates by perturbing a single-objective optimum
    with Gaussian noise.

    Each parent is sampled as:
        x_parent = x_star + N(0, noise_scale * (xu - xl))

    The noise is scaled by the per-dimension range so that the perturbation
    is proportional to the local search space width, rather than being an
    absolute offset that ignores parameter scale differences.

    Parameters
    ----------
    x_star      : np.ndarray (n_var,), the verified single-objective optimum
    n_parents   : number of parent candidates to generate
    noise_scale : fraction of per-dimension range used as noise std
                  (default 0.05 = 5% of each dimension's range)
    xl          : lower bounds for clipping (recommended: pass substrate bounds)
    xu          : upper bounds for clipping (recommended: pass substrate bounds)
    rng         : numpy random Generator (created from seed 0 if not provided)

    Returns
    -------
    parents : np.ndarray of shape (n_parents, n_var)
    """
    rng    = rng or np.random.default_rng(0)
    x_star = np.asarray(x_star, dtype=np.float64)
    n_var  = x_star.shape[0]

    if xl is not None and xu is not None:
        xl  = np.asarray(xl, dtype=np.float64)
        xu  = np.asarray(xu, dtype=np.float64)
        std = noise_scale * (xu - xl)
    else:
        std = noise_scale * (np.abs(x_star) + 1.0)

    noise   = rng.normal(loc=0.0, scale=std, size=(n_parents, n_var))
    parents = x_star[None, :] + noise

    if xl is not None and xu is not None:
        parents = np.clip(parents, xl, xu)

    return parents
