import substrates
import foundation_models
from eval_fn import encode_prompts, build_rollout_fn, get_substrate_bounds
from aurora_qd import AuroraArchive, AuroraManager
from problems.pymooproblem_definition import MOO_ASALProblem

def build_moo_problem(args):
    prompts = args.prompts.split(";")

    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    if args.rollout_steps is not None:
        substrate.rollout_steps = args.rollout_steps

    rollout_fn = build_rollout_fn(substrate, fm, time_sampling=args.time_sampling)
    z_txt_list = encode_prompts(fm, prompts)

    xl, xu = get_substrate_bounds(substrate)

    aurora_archive = AuroraArchive(max_size=5000)
    aurora_manager = AuroraManager(
        archive=aurora_archive,
        k=10,
        normalize_bd=True,
    )

    problem = MOO_ASALProblem(
        rollout_fn=rollout_fn,
        z_txt_list=z_txt_list,
        xl=xl,
        xu=xu,
        base_seed=args.seed,
        n_evals=args.n_evals,
        novelty_evaluator=aurora_manager,
    )

    return problem, aurora_archive