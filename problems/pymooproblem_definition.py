import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from pymoo.core.problem import Problem
import asal_metrics


class MOO_ASALProblem(Problem):
    """
    General-purpose ASAL multi-objective problem definition.

    Each objective = calc_supervised_target_score(z, z_txt_i) for prompt i.
    Scores are already negated inside calc_supervised_target_score (returns
    negative similarity), so pymoo minimizes them correctly.

    Parameters
    ----------
    rollout_fn   : jitted callable (rng, x_jax) -> {'z': (T, D)}
    z_txt_list   : list of jnp.ndarray, each shape (1, D) — one per prompt
    xl, xu       : np.ndarray (n_var,) — fixed search bounds
    score_fn     : scoring function, defaults to calc_supervised_target_score
    base_seed    : int, global salt for rng derivation
    n_evals      : int, number of independent rollouts to average per evaluation
    novelty_evaluator : optional, used for novelty objective
    """

    def __init__(
        self,
        rollout_fn,
        z_txt_list,
        xl,
        xu,
        score_fn=None,
        base_seed=0,
        n_evals=1,
        novelty_evaluator=None,
    ):
        self.rollout_fn = rollout_fn
        self.z_txt_list = z_txt_list
        self.base_seed = base_seed
        self.score_fn = score_fn or asal_metrics.calc_supervised_target_score
        self.n_evals = n_evals
        self.novelty_evaluator = novelty_evaluator

        # 生成 objective metadata
        prompt_obj_names = [f"prompt_{i}" for i in range(len(z_txt_list))]
        if novelty_evaluator is not None:
            self.objective_names = prompt_obj_names + ["novelty"]
            self.objective_roles = ["similarity"] * len(z_txt_list) + ["novelty"]
        else:
            self.objective_names = prompt_obj_names
            self.objective_roles = ["similarity"] * len(z_txt_list)

        # jit vectorized functions
        self._batch_rollout = jax.jit(
            jax.vmap(self.rollout_fn, in_axes=(0, 0))
        )
        self._batch_score = jax.jit(
            jax.vmap(self.score_fn, in_axes=(0, None))
        )

        super().__init__(
            n_var=len(xl),
            n_obj=len(z_txt_list) + (1 if novelty_evaluator is not None else 0),
            n_ieq_constr=0,
            xl=np.asarray(xl, dtype=np.float64),
            xu=np.asarray(xu, dtype=np.float64),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        X_jax = jnp.asarray(X, dtype=jnp.float32)

        # Base prompt objectives
        scores = np.zeros((pop_size, len(self.z_txt_list)), dtype=np.float64)
        z_accum = None

        for i in range(self.n_evals):
            rngs = self._derive_rng_keys(X, salt=i)
            data = self._batch_rollout(rngs, X_jax)
            z = data["z"]

            if isinstance(z, (tuple, list)):
                z = z[0]

            z_np = np.asarray(z, dtype=np.float64)
            if z_accum is None:
                z_accum = z_np
            else:
                z_accum += z_np

            for j, z_txt in enumerate(self.z_txt_list):
                batch_scores = self._batch_score(z, z_txt)
                scores[:, j] += np.asarray(batch_scores, dtype=np.float64)

        base_F = scores / self.n_evals
        z_mean = z_accum / self.n_evals

        if self.novelty_evaluator is not None:
            novelty = self.novelty_evaluator.score_from_z(z_mean)
            out["F"] = np.concatenate([base_F, -novelty[:, None]], axis=1)
            out["BD"] = self.novelty_evaluator.compute_bd(z_mean)
        else:
            out["F"] = base_F

        # Expose averaged sensory data for possible external use
        out["Z"] = z_mean

    def _derive_rng_keys(self, X: np.ndarray, salt: int = 0) -> jax.Array:
        """
        Derive deterministic rng keys for the entire population.
        SHA256 avoids hash collisions; salt enables multi-rollout averaging.
        """
        keys = []
        for x in X:
            digest = hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).digest()
            seed = (
                self.base_seed
                ^ int.from_bytes(digest[:4], "little")
                ^ ((salt * 2654435761) & 0xFFFFFFFF)
            ) % (2 ** 31)
            keys.append(jax.random.PRNGKey(seed))
        return jnp.stack(keys)  # (pop_size, 2)