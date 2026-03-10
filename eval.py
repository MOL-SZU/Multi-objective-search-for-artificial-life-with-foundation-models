import jax
import jax.numpy as jnp

def get_batch_loss_fn(rollout_fn, fm, prompts):
    z_txt_list = [fm.embed_txt([p]) for p in prompts]

    def evaluate_single(rng, params):
        rollout_data = rollout_fn(rng, params)
        z = rollout_data['z']  # (T, D)
        
        # --- 核心排查：强制 L2 归一化 ---
        # 很多时候 FM 返回的特征在 JAX 运算中会丢失归一化属性
        # 导致点积结果偏小。强制归一化确保我们算的是纯粹的 Cosine Similarity
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
        
        raw_sims = []
        for z_t in z_txt_list:
            # 同样对文本特征做归一化验证
            z_t_norm = z_t / (jnp.linalg.norm(z_t, axis=-1, keepdims=True) + 1e-6)
            
            # 计算相似度序列
            cos_sim_path = jnp.dot(z, z_t_norm.T).squeeze() # (T,)
            
            # 改进：取最后 10% 帧的平均值，或者全过程的最大值
            # 这样可以避开 Lenia 发育初期的“无意义阶段”
            final_stage_sim = jnp.max(cos_sim_path)
            raw_sims.append(final_stage_sim)
        
        scores = jnp.array(raw_sims)
        return scores, {"scores": scores}

    def evaluate_population(rng, params_pop):
        v_eval = jax.vmap(evaluate_single, in_axes=(None, 0))
        return v_eval(rng, params_pop)

    return jax.jit(evaluate_population)
