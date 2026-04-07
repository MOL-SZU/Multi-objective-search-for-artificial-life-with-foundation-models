import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import argparse
import util
from pymoo.optimize import minimize

from configs.loader import load_yaml_config, flatten_config
from problems.factory import build_moo_problem
from algorithms.factory import build_moo_algorithm, build_resume_population
from runtime.checkpoint import (
    load_latest_checkpoint,
    get_archive_len,
    validate_checkpoint,
)
from runtime.callback import ArchiveAndCheckpointCallback


parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default=None, help="path to yaml config")

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
group.add_argument("--algorithm", type=str, default="nsga2", help="multi-objective algorithm: nsga2 or moead")
group.add_argument("--pop_size", type=int, default=64, help="population size for NSGA-II")
group.add_argument("--n_gen", type=int, default=20000, help="TOTAL number of generations to run")
group.add_argument("--checkpoint_dir", type=str, default=None, help="directory for checkpoints (enables resume)")
group.add_argument("--save_every", type=int, default=100, help="save checkpoint every N generations")
group.add_argument("--n_neighbors", type=int, default=20, help="number of neighbors for MOEA/D")
group.add_argument("--decomposition", type=str, default="pbi", help="MOEA/D decomposition method")
group.add_argument("--n_partitions", type=int, default=12, help="number of reference-direction partitions for MOEA/D")


def parse_args(*args, **kwargs):
    # 1) 先解析 config path
    preliminary = parser.parse_known_args(*args, **kwargs)[0]

    # 2) load yaml config if provided
    yaml_cfg = {}
    if getattr(preliminary, "config", None):
        yaml_cfg = load_yaml_config(preliminary.config) or {}
    flat_cfg = flatten_config(yaml_cfg) if yaml_cfg else {}

    # 3) parse all command line args normally
    args = parser.parse_args(*args, **kwargs)

    # 4) override parser defaults with YAML when CLI did NOT explicitly specify
    for key, val in flat_cfg.items():
        # 当命令行没有指定时，才用 yaml 覆盖
        if f"--{key}" not in sys.argv:
            setattr(args, key, val)

    # 5) unify string "none" to None
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)

    return args


def main(args):
    # ensure time_sampling at least = number of prompts
    prompts = args.prompts.split(";")
    if args.time_sampling < len(prompts):
        args.time_sampling = len(prompts)

    print(args)

    problem, aurora_archive = build_moo_problem(args)
    print(f"[Setup] {len(prompts)} prompts encoded.")
    print(f"[Setup] Problem: n_var={problem.n_var}, n_obj={problem.n_obj}")

    ckpt = load_latest_checkpoint(args.checkpoint_dir)
    start_gen = ckpt["gen"] if ckpt is not None else 0
    remaining_gen = args.n_gen - start_gen

    if remaining_gen <= 0:
        print(f"[Done] All {args.n_gen} generations already completed.")
        return

    initial_archive_size = get_archive_len(args.save_dir)

    if ckpt is not None:
        resume_pop = build_resume_population(ckpt)
        algorithm, algo_meta = build_moo_algorithm(args, problem, resume_pop=resume_pop)
        validate_checkpoint(ckpt, problem, algo_meta)
        print(f"[Resume] Resuming from global gen {start_gen}, {remaining_gen} gens remaining.")
        print(f"[Resume] Existing archive size: {initial_archive_size}")
    else:
        resume_pop = None
        algorithm, algo_meta = build_moo_algorithm(args, problem, resume_pop=None)
        print("[Resume] Starting a fresh run.")

    callback = ArchiveAndCheckpointCallback(
        save_dir=args.save_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        start_gen=start_gen,
        initial_archive_size=initial_archive_size,
        novelty_archive=aurora_archive,
        algo_name=algo_meta["name"],
        front_mode=algo_meta["front_mode"],
        args=args,
        problem=problem,
        algo_meta=algo_meta,
    )

    print(
        f"\n[Run] {algo_meta['name']} | pop={algo_meta['effective_pop_size']} "
        f"| remaining_gen={remaining_gen} | total_target_gen={args.n_gen} "
        f"| n_obj={problem.n_obj}"
    )
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
        util.save_pkl(args.save_dir, "X_nd", res.X)
        util.save_pkl(args.save_dir, "F_nd", res.F)

    print("\n[Done] Optimisation complete.")
    print(f"       Pareto/front size : {len(res.F)}")
    print(f"       Total archive size: {callback.archive_size}")
    print(">>> algorithm arg:", args.algorithm)

    obj_names = getattr(problem, "objective_names", [f"obj_{i}" for i in range(res.F.shape[1])])
    for i, row in enumerate(-res.F):
        labels = [f"{name}={val:.4f}" for name, val in zip(obj_names, row)]
        print(f"         Solution {i:03d}: {' | '.join(labels)}")


if __name__ == '__main__':
    main(parse_args())