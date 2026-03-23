import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from pymoo.core.problem import ElementwiseProblem
import asal_metrics


class MOO_ASALProblem(ElementwiseProblem):
    """
    General-purpose ASAL multi-objective problem definition.

    Responsibilities of this class:
      - Accepts an externally built and jitted rollout_fn
      - Accepts pre-computed text embeddings (z_txt_list)
      - Accepts externally computed search bounds (xl, xu)
      - Sole responsibility: x -> rollout -> score -> return F
    """

    def __init__(
        self,
        rollout_fn,      # externally jitted callable: (rng, x_jax) -> {'z': (T, D)}
        z_txt_list,      # list of jnp.ndarray, each shape (T2, D)
        xl,              # np.ndarray (n_var,), lower bounds passed from outside
        xu,              # np.ndarray (n_var,), upper bounds passed from outside
        score_fn=None,   # scoring function, defaults to calc_supervised_target_score
        base_seed=0,     # global salt for rng key derivation
    ):
        self.rollout_fn = rollout_fn
        self.z_txt_list = z_txt_list
        self.base_seed  = base_seed
        self.score_fn   = score_fn or asal_metrics.calc_supervised_target_score

        super().__init__(
            n_var=len(xl),
            n_obj=len(z_txt_list),
            n_ieq_constr=0,
            xl=np.asarray(xl, dtype=np.float64),
            xu=np.asarray(xu, dtype=np.float64),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        rng   = self._derive_rng_key(x)
        x_jax = jnp.asarray(x, dtype=jnp.float32)

        data  = self.rollout_fn(rng, x_jax)
        z     = data["z"]   # shape: (T, D)

        out["F"] = np.array(
            [float(self.score_fn(z, z_txt)) for z_txt in self.z_txt_list],
            dtype=np.float64,
        )

    def _derive_rng_key(self, x: np.ndarray) -> jax.Array:
        # SHA256 fingerprint eliminates collision risk from simple summation
        digest = hashlib.sha256(
            np.asarray(x, dtype=np.float32).tobytes()
        ).digest()
        seed = (self.base_seed ^ int.from_bytes(digest[:4], "little")) % (2 ** 31)
        return jax.random.PRNGKey(seed)