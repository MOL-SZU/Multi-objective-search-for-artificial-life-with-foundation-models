from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff


def build_resume_population(ckpt):
    if ckpt is None:
        return None
    return Population.new("X", ckpt["pop_X"], "F", ckpt["pop_F"])


def build_nsga2(args, resume_pop=None):
    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=args.crossover_prob if hasattr(args, "crossover_prob") else 0.9,
                      eta=args.crossover_eta if hasattr(args, "crossover_eta") else 10),
        mutation=PM(prob=args.mutation_prob if hasattr(args, "mutation_prob") else 0.2,
                    eta=args.mutation_eta if hasattr(args, "mutation_eta") else 10),
        eliminate_duplicates=True,
    )

    if resume_pop is not None:
        algorithm.initialization.sampling = resume_pop

    algo_meta = {
        "name": "NSGA-II",
        "front_mode": "opt",
        "use_nsga2_log_format": True,
        "effective_pop_size": args.pop_size,
    }
    return algorithm, algo_meta


def build_moead(args, problem, resume_pop=None):
    ref_dirs = get_reference_directions(
        "das-dennis",
        problem.n_obj,
        n_partitions=args.n_partitions,
    )

    # 解析 decomposition 字符串生成实际 decomposition 对象
    decomp_str = args.decomposition.lower() if isinstance(args.decomposition, str) else None
    if decomp_str in {"pbi"}:
        decomp_obj = PBI()
    elif decomp_str in {"tchebi", "tchebicheff"}:
        decomp_obj = Tchebicheff()
    else:
        # 默认让 pymoo 自己设定
        decomp_obj = None

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=args.n_neighbors,
        decomposition=decomp_obj,  # 传递实际实例
        sampling=FloatRandomSampling(),
        crossover=SBX(
            prob=args.crossover_prob if hasattr(args, "crossover_prob") else 0.9,
            eta=args.crossover_eta if hasattr(args, "crossover_eta") else 10,
        ),
        mutation=PM(
            prob=args.mutation_prob if hasattr(args, "mutation_prob") else 0.2,
            eta=args.mutation_eta if hasattr(args, "mutation_eta") else 10,
        ),
    )

    if resume_pop is not None:
        algorithm.initialization.sampling = resume_pop

    algo_meta = {
        "name": "MOEA/D",
        "front_mode": "nondominated_from_pop",
        "use_nsga2_log_format": False,
        "effective_pop_size": len(ref_dirs),
    }
    return algorithm, algo_meta


def build_moo_algorithm(args, problem, resume_pop=None):
    algo_name = args.algorithm.lower()

    if algo_name == "nsga2":
        return build_nsga2(args, resume_pop=resume_pop)

    if algo_name == "moead":
        return build_moead(args, problem, resume_pop=resume_pop)

    raise ValueError(f"Unsupported algorithm: {args.algorithm}")