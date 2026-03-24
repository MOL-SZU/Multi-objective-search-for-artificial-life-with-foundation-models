import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from rollout import rollout_simulation


def encode_prompts(fm, prompts: list[str]) -> list[jnp.ndarray]:
    """
    Pre-compute text embeddings for all prompts.
    Returns a list of jnp.ndarray, each shape (1, D).
    """
    z_txt_list = []
    for p in prompts:
        z = fm.embed_txt([p])
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        z_txt_list.append(z)
    return z_txt_list


def build_rollout_fn(substrate, fm, time_sampling=8):
    """
    Build and jit the rollout function.

    Parameters
    ----------
    substrate     : ASAL substrate instance (FlattenSubstrateParameters or raw)
    fm            : foundation model instance
    time_sampling : number of frames sampled evenly across the trajectory
    """
    inner_sub     = getattr(substrate, 'substrate', substrate)
    rollout_steps = getattr(inner_sub, 'rollout_steps', 256)
    # Always use 224 for CLIP/DINO compatibility, not grid_size
    img_size      = 224

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


_SUBSTRATE_BOUNDS = {
    # Lenia: offset in logit space around base_params; sigmoid(base+offset) in (0,1)
    "lenia":      (-8.0,  8.0),
    # Boids: neural network weights
    "boids":      (-3.0,  3.0),
    # ParticleLife: raw normal params fed through sigmoid/exp transforms
    "plife":      (-5.0,  5.0),
    "plife_plus": (-3.0,  3.0),
    # ParticleLenia: 6 log-scale params
    "plenia":     (-5.0,  5.0),
    # NCA variants: network weights
    "nca_d1":     (-3.0,  3.0),
    "nca_d3":     (-3.0,  3.0),
    # DNCA: network weights + init logits
    "dnca":       (-5.0,  5.0),
    # GameOfLife: integer rule encoded as float
    "gol":        ( 0.0,  2.0**18 - 1),
}


def get_substrate_bounds(substrate) -> tuple[np.ndarray, np.ndarray]:
    """
    Return fixed per-dimension bounds based on substrate type.
    Falls back to [-10, 10] if substrate name is unknown.

    Parameters
    ----------
    substrate : FlattenSubstrateParameters or raw substrate with .name and .n_params

    Returns
    -------
    xl, xu : np.ndarray of shape (n_params,), or (None, None) if n_params unknown
    """
    name     = getattr(substrate, 'name', None)
    n_params = getattr(substrate, 'n_params', None)

    if n_params is None:
        raise ValueError("[Bounds] n_params unknown.")

    if name not in _SUBSTRATE_BOUNDS:
        raise ValueError(
            f"[Bounds] Unknown substrate '{name}'. "
            "Please define it in _SUBSTRATE_BOUNDS."
        )

    lo, hi = _SUBSTRATE_BOUNDS.get(name, (-10.0, 10.0))
    xl = np.full(n_params, lo, dtype=np.float64)
    xu = np.full(n_params, hi, dtype=np.float64)

    print(f"[Bounds] substrate='{name}' | n_params={n_params} | "
          f"fixed bounds=[{lo}, {hi}]")
    return xl, xu
