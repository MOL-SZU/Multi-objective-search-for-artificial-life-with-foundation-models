import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util

# 参数解析器
parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='lenia', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use")
group.add_argument("--time_sampling", type=int, default=1, help="images to render during rollout")
group.add_argument("--prompts", type=str, default="a caterpillar", help="prompts to optimize")
group.add_argument("--coef_prompt", type=float, default=1., help="coefficient for ASAL prompt loss")
group.add_argument("--coef_softmax", type=float, default=0., help="coefficient for softmax loss")
group.add_argument("--coef_oe", type=float, default=0., help="coefficient for ASAL open-endedness loss")

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states")
group.add_argument("--pop_size", type=int, default=16, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")

def setup_evaluator(args):
    """供外部调用的初始化逻辑"""
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    
    rollout_fn = partial(
        rollout_simulation, 
        s0=None, substrate=substrate, fm=fm, 
        rollout_steps=args.rollout_steps or substrate.rollout_steps, 
        time_sampling=(args.time_sampling, True), 
        img_size=224, return_state=False
    )
    return rollout_fn, fm, substrate

def parse_args(*args, **kwargs):
    parsed_args = parser.parse_args(*args, **kwargs)
    for k, v in vars(parsed_args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(parsed_args, k, None)
    return parsed_args

def optimize(args):
    """核心优化函数：寻找单目标种子"""
    prompts = args.prompts.split(";")
    if args.time_sampling < len(prompts):
        args.time_sampling = len(prompts)
    
    rollout_fn, fm, substrate = setup_evaluator(args)
    z_txt = fm.embed_txt(prompts)

    rng = jax.random.PRNGKey(args.seed)
    # 使用 Sep-CMA-ES 寻找最优参数
    strategy = evosax.Sep_CMA_ES(popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma)
    es_params = strategy.default_params
    rng, _rng = split(rng)
    es_state = strategy.initialize(_rng, es_params)

    def calc_loss(rng, params):
        rollout_data = rollout_fn(rng, params)
        z = rollout_data['z']
        loss_prompt = asal_metrics.calc_supervised_target_score(z, z_txt)
        loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
        loss_oe = asal_metrics.calc_open_endedness_score(z)

        loss = loss_prompt * args.coef_prompt + loss_softmax * args.coef_softmax + loss_oe * args.coef_oe
        loss_dict = dict(loss=loss, loss_prompt=loss_prompt, loss_oe=loss_oe)
        return loss, loss_dict

    @jax.jit
    def do_iter(es_state, rng):
        rng, _rng = split(rng)
        params, next_es_state = strategy.ask(_rng, es_state, es_params)
        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))
        rng, _rng = split(rng)
        loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params)
        loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict))
        next_es_state = strategy.tell(params, loss, next_es_state, es_params)
        return next_es_state, dict(best_loss=next_es_state.best_fitness)

    # 进度条展示
    data_history = []
    pbar = tqdm(range(args.n_iters), desc=f"Seeding: {args.prompts}")
    for i_iter in pbar:
        rng, _rng = split(rng)
        es_state, di = do_iter(es_state, _rng)
        data_history.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())
        
        # 结果定期保存
        if args.save_dir and (i_iter % (args.n_iters // 10) == 0 or i_iter == args.n_iters - 1):
            os.makedirs(args.save_dir, exist_ok=True)
            best_res = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best_res)

    # 返回 NumPy 格式的最佳成员，以便注入 MOO
    return np.array(es_state.best_member), es_state.best_fitness

def main(args):
    optimize(args)

if __name__ == '__main__':
    main(parse_args())
