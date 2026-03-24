import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
from functools import partial
from tqdm.auto import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

import substrates
import foundation_models
from rollout import rollout_simulation
from problems.pymooproblem_definition import MOO_ASALProblem
from eval_fn import encode_prompts, build_rollout_fn, get_substrate_bounds
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default='lenia', help="name of the substrate")
group.add_argument("--rollout_steps", type=int, default=None, help="number of rollout timesteps, leave None for the default of the substrate")

group = parser.add_argument_group("evaluation")
group.add_argument("--foundation_model", type=str, default="clip", help="the foundation model to use")
group.add_argument("--time_sampling", type=int, default=8, help="number of images to render during one simulation rollout")
group.add_argument("--prompts", type=str, default="a butterfly;a caterpillar", help="prompts to optimize for separated by ';'")
group.add_argument("--n_evals", type=int, default=1, help="number of independent rollouts to average per evaluation")

group = parser.add_argument_group("optimization")
group.add_argument("--pop_size", type=int, default=64, help="population size for NSGA-II")
group.add_argument("--n_gen", type=int, default=20000, help="number of generations to run")
group.add_argument("--checkpoint_dir", type=str, default=None, help="directory for checkpoints (enables resume)")
group.add_argument("--save_every", type=int, default=100, help="save checkpoint every N generations")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args


class ArchiveAndCheckpointCallback(Callback):
    def __init__(self, save_dir, checkpoint_dir, save_every=100, initial_archive=None):
        super().__init__()
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.archive = initial_archive or []
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        X_all = pop.get("X")
        F_all = pop.get("F")

        for x, f in zip(X_all, F_all):
            self.archive.append({"gen": gen, "X": x.copy(), "F": f.copy()})

        if self.save_dir is not None:
            util.save_pkl(self.save_dir, "archive", self.archive)

        if gen % self.save_every == 0 and self.checkpoint_dir is not None:
            ckpt = {"gen": gen, "pop_X": X_all.copy(), "pop_F": F_all.copy()}
            util.save_pkl(self.checkpoint_dir, f"ckpt_gen{gen:04d}", ckpt)
            util.save_pkl(self.checkpoint_dir, "latest", ckpt)

            if self.checkpoint_dir is not None:
                ckpt = {"gen": gen, "pop_X": X_all.copy(), "pop_F": F_all.copy()}
                util.save_pkl(self.checkpoint_dir, f"ckpt_gen{gen:04d}", ckpt)
                util.save_pkl(self.checkpoint_dir, "latest", ckpt)

        # Pareto 前沿上的解
        pareto_F = algorithm.opt.get("F")           # (n_pareto, n_obj)
        pareto_sim = -pareto_F                       # 转为 similarity，越大越好
        best_sim = pareto_sim.max(axis=0)            # 每个目标的最优 similarity
        mean_sim = pareto_sim.mean(axis=0)           # Pareto 前沿的平均 similarity
    
        print(
            f"[NSGA-II | Gen {gen:04d}] "
            f"pareto={len(pareto_F)} | "
            f"archive={len(self.archive)} | "
            f"best_sim=[{', '.join(f'{v:.4f}' for v in best_sim)}] | "
            f"mean_sim=[{', '.join(f'{v:.4f}' for v in mean_sim)}]",
            flush=True
        )


def load_latest_checkpoint(checkpoint_dir):
    if checkpoint_dir is None:
        return None
    latest_path = os.path.join(checkpoint_dir, "latest.pkl")
    if not os.path.exists(latest_path):
        return None
    ckpt = util.load_pkl(checkpoint_dir, "latest")
    print(f"[Resume] Checkpoint at gen {ckpt['gen']} ({len(ckpt['pop_X'])} individuals)")
    return ckpt


def main(args):
    prompts = args.prompts.split(";")
    if args.time_sampling < len(prompts):
        args.time_sampling = len(prompts)
    print(args)

    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)
    if args.rollout_steps is not None:
        substrate.rollout_steps = args.rollout_steps

    rollout_fn = build_rollout_fn(substrate, fm, time_sampling=args.time_sampling)
    z_txt_list = encode_prompts(fm, prompts)
    print(f"[Setup] {len(z_txt_list)} prompts encoded, dim={z_txt_list[0].shape[-1]}")

    xl, xu = get_substrate_bounds(substrate)
    if xl is None:
        raise RuntimeError(f"Could not determine bounds for substrate '{args.substrate}'.")

    problem = MOO_ASALProblem(
        rollout_fn=rollout_fn,
        z_txt_list=z_txt_list,
        xl=xl,
        xu=xu,
        base_seed=args.seed,
        n_evals=args.n_evals,
    )
    print(f"[Setup] Problem: n_var={problem.n_var}, n_obj={problem.n_obj}")

    ckpt = load_latest_checkpoint(args.checkpoint_dir)
    start_gen = ckpt["gen"] if ckpt is not None else 0
    remaining_gen = args.n_gen - start_gen

    if remaining_gen <= 0:
        print(f"[Done] All {args.n_gen} generations already completed.")
        return

    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=False,
    )

    if ckpt is not None:
        resume_pop = Population.new("X", ckpt["pop_X"], "F", ckpt["pop_F"])
        algorithm.initialization.sampling = resume_pop
        prior_archive = util.load_pkl(args.save_dir, "archive") if args.save_dir else None
        print(f"[Resume] Resuming from gen {start_gen}, {remaining_gen} gens remaining.")
    else:
        prior_archive = None

    callback = ArchiveAndCheckpointCallback(
        save_dir=args.save_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        initial_archive=prior_archive,
    )

    print(f"\n[Run] NSGA-II | pop={args.pop_size} | n_gen={remaining_gen} | n_obj={problem.n_obj}")
    print("=" * 60)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", remaining_gen),
        seed=args.seed,
        callback=callback,
        verbose=False,
    )

    if args.save_dir is not None:
        util.save_pkl(args.save_dir, "pareto_X", res.X)
        util.save_pkl(args.save_dir, "pareto_F", res.F)
        util.save_pkl(args.save_dir, "archive", callback.archive)

    print(f"\n[Done] Optimisation complete.")
    print(f"       Pareto front size : {len(res.F)}")
    print(f"       Total archive size: {len(callback.archive)}")
    print(f"       Similarity scores (higher is better):")
    for i, row in enumerate(-res.F):
        labels = [f"prompt_{j}={v:.4f}" for j, v in enumerate(row)]
        print(f"         Solution {i:03d}: {' | '.join(labels)}")


if __name__ == '__main__':
    main(parse_args())

# python main_opt_moo.py \
#     --seed 0 \                        # 随机种子
#     --save_dir ./results \            # 结果保存目录 (pareto_X, pareto_F, archive)
#     --substrate lenia \               # substrate名称
#     --foundation_model clip \         # 基础模型
#     --time_sampling 8 \               # 每次rollout采样的帧数
#     --prompts "a butterfly;a caterpillar" \  # 多目标prompt，用;分隔
#     --n_evals 1 \                     # 每个解评估的rollout次数
#     --pop_size 64 \                   # NSGA-II种群大小
#     --n_gen 20000 \                   # 总代数
#     --checkpoint_dir ./checkpoints \  # checkpoint目录 (支持断点续跑)
#     --save_every 100                  # 每100代保存一次checkpoint
#python main_opt_moo.py --seed 0 --save_dir ./results --substrate lenia --foundation_model clip --time_sampling 8 --prompts "a butterfly;a caterpillar" --n_evals 1 --pop_size 64 --n_gen 20000 --checkpoint_dir ./checkpoints --save_every 100