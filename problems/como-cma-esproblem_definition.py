import numpy as np
import jax
import jax.numpy as jnp
import comocma

class ASALProblem:
    def __init__(self, rollout_fn, z_txt_list, xl, xu, n_evals=1):
        self.rollout_fn = rollout_fn
        self.z_txt_all = jnp.stack(z_txt_list)  # (n_obj, D)
        self.xl = jnp.asarray(xl)
        self.xu = jnp.asarray(xu)
        self.n_evals = n_evals
        self.n_obj = len(z_txt_list)
        
        # 定义核心评估逻辑：输入单个 x 和单个 target，输出 scalar score
        def single_score_fn(rng, x, target):
            # 假设 rollout_fn 返回 {'z': (T, D)}
            res = rollout_fn(rng, x)
            # 这里调用你的指标计算逻辑（确保是最小化：相似度越高，得分越负）
            # 示例：-cosine_similarity(res['z'][-1], target)
            return jnp.dot(res['z'][-1], target) / (jnp.linalg.norm(res['z'][-1]) * jnp.linalg.norm(target) + 1e-6) * -1.0

        # 对种群 vmap (in_axes=0)，对目标 vmap (in_axes=None, ..., 0)
        self._batch_eval = jax.jit(jax.vmap(
            jax.vmap(single_score_fn, in_axes=(None, 0, None)), # 针对种群
            in_axes=(None, None, 0) # 针对目标
        ))

    def evaluate(self, X, generation_seed):
        # X: (pop_size, n_var)
        X_jax = jnp.asarray(X)
        pop_size = X.shape[0]
        
        # 简单高效的 RNG 生成
        rng = jax.random.PRNGKey(generation_seed)
        
        # 结果形状: (n_obj, pop_size) -> 转置为 (pop_size, n_obj)
        scores = self._batch_eval(rng, X_jax, self.z_txt_all)
        return np.array(scores.T)