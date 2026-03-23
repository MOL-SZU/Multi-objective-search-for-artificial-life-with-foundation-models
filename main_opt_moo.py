import os
import pickle
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

from problems.pymooproblem_definition import MOO_ASALProblem
from eval_fn import (
    encode_prompts,
    build_rollout_fn,
    get_substrate_bounds,
    compute_bounds_from_seeds,
    generate_parents_from_optimum,
)


# =============================================================================
# Configuration
# =============================================================================

PROMPTS = [
    "a butterfly",
    "a caterpillar",
]

# --- Extreme solutions (one per prompt, manually verified) ---
EXTREME_PATHS = [
    "\seed_butterfly\best.pkl",
    "\seed_caterpillar\best.pkl",
]

# --- Parent generation from single-objective optima ---
N_PARENTS_PER_PROMPT = 9      # 9 parents x 2 prompts = 18 parents total
NOISE_SCALE          = 0.05   # 5% of per-dimension range as Gaussian noise std

# --- Population composition (must sum to POP_SIZE) ---
# 2 extremes + 18 parents + 44 random = 64
POP_SIZE   = 64
N_EXTREMES = len(EXTREME_PATHS)
N_PARENTS  = N_PARENTS_PER_PROMPT * len(EXTREME_PATHS)
N_RANDOM   = POP_SIZE - N_EXTREMES - N_PARENTS

assert N_RANDOM >= 0, (
    f"Population oversubscribed: {N_EXTREMES} extremes + {N_PARENTS} parents "
    f"> {POP_SIZE} pop_size. Reduce N_PARENTS_PER_PROMPT or increase POP_SIZE."
)

# --- Multi-objective optimisation ---
N_GEN         = 20000
BASE_SEED     = 42
TIME_SAMPLING = 8

# --- Bound expansion margins ---
DYN_MARGIN  = 0.5
INIT_MARGIN = 1.0

# --- Output paths ---
CHECKPOINT_DIR = "checkpoints"
ARCHIVE_PATH   = "archive.pkl"
RESULT_DIR     = "results"


# =============================================================================
# Loaders
# =============================================================================

def load_extremes(extreme_paths: list[str]) -> np.ndarray:
    """
    Load and stack single-objective optimal solutions from the given paths.
    Hard-fails if any path is missing to prevent silent fallback
    to uninformed bounds.

    Returns
    -------
    extremes : np.ndarray of shape (n_prompts, n_var)
    """
    x_stars = []
    for i, path in enumerate(extreme_paths):
        assert os.path.exists(path), (
            f"Extreme solution for prompt {i} not found at '{path}'.\n"
            f"Run single_obj_search.py first, verify manually, "
            f"then update EXTREME_PATHS."
        )
        x = np.load(path)
        x_stars.append(x)
        print(f"[Load] Extreme {i} loaded from '{path}' | shape: {x.shape}")
    return np.stack(x_stars)   # (n_prompts, n_var)


# =============================================================================
# Archive and checkpoint callback
# =============================================================================

class ArchiveAndCheckpointCallback(Callback):
    """
    After every generation:
      - Appends every individual in the current population to the global archive.
      - Saves the archive to disk.
      - Saves a generation checkpoint for resume support.

    Archive entry format:
        {'gen': int, 'X': np.ndarray (n_var,), 'F': np.ndarray (n_obj,)}
    """

    def __init__(self, archive_path: str, checkpoint_dir: str,
                 initial_archive: list | None = None):
        super().__init__()
        self.archive_path   = archive_path
        self.checkpoint_dir = checkpoint_dir
        self.archive        = initial_archive or []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def notify(self, algorithm):
        gen   = algorithm.n_gen
        pop   = algorithm.pop
        X_all = pop.get("X")
        F_all = pop.get("F")

        for x, f in zip(X_all, F_all):
            self.archive.append({"gen": gen, "X": x.copy(), "F": f.copy()})

        with open(self.archive_path, "wb") as fp:
            pickle.dump(self.archive, fp)

        ckpt = {"gen": gen, "pop_X": X_all.copy(), "pop_F": F_all.copy()}

        ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_gen{gen:04d}.pkl")
        with open(ckpt_path, "wb") as fp:
            pickle.dump(ckpt, fp)

        latest_path = os.path.join(self.checkpoint_dir, "latest.pkl")
        with open(latest_path, "wb") as fp:
            pickle.dump(ckpt, fp)

        print(
            f"[NSGA-II | Gen {gen:04d}] "
            f"n_eval={algorithm.evaluator.n_eval} | "
            f"pareto={len(algorithm.opt)} | "
            f"archive={len(self.archive)}"
        )


# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_latest_checkpoint(checkpoint_dir: str) -> dict | None:
    latest_path = os.path.join(checkpoint_dir, "latest.pkl")
    if not os.path.exists(latest_path):
        return None
    with open(latest_path, "rb") as fp:
        ckpt = pickle.load(fp)
    print(f"[Resume] Loaded checkpoint at generation {ckpt['gen']} "
          f"({len(ckpt['pop_X'])} individuals)")
    return ckpt


def load_archive(archive_path: str) -> list | None:
    if not os.path.exists(archive_path):
        return None
    with open(archive_path, "rb") as fp:
        archive = pickle.load(fp)
    print(f"[Resume] Loaded existing archive ({len(archive)} entries)")
    return archive


# =============================================================================
# Initial population
# =============================================================================

def build_initial_population(
    problem: MOO_ASALProblem,
    extremes: np.ndarray,
    parents: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
) -> Population:
    """
    Compose the initial population from three sources:
      1. Single-objective extreme solutions  (2)  — Pareto corner anchors
      2. Parent pool from Gaussian expansion (18) — neighbourhood of optima
      3. Random fill                         (44) — diversity

    All individuals are clipped to [xl, xu] and evaluated before
    being handed to NSGA-II.
    """
    random_X = rng.uniform(
        problem.xl, problem.xu,
        size=(n_random, problem.n_var)
    )

    all_X = np.vstack([
        np.atleast_2d(extremes),   # (2,  n_var)
        np.atleast_2d(parents),    # (18, n_var)
        random_X,                  # (44, n_var)
    ])

    all_X = np.clip(all_X, problem.xl, problem.xu)

    print(
        f"[Init] Population composed: "
        f"{len(extremes)} extreme(s) + "
        f"{len(parents)} parent(s) + "
        f"{n_random} random = {len(all_X)} total"
    )

    pop = Population.new("X", all_X)
    Evaluator().eval(problem, pop)
    return pop


# =============================================================================
# Main
# =============================================================================

def main():

    # ── 1. Substrate and foundation model ────────────────────────────────────
    from substrates.lenia import Substrate, FlattenSubstrateParameters
    from foundation_models.clip import FoundationModel

    substrate = Substrate()
    fm        = FoundationModel()

    # ── 2. Text embeddings ────────────────────────────────────────────────────
    z_txt_list = encode_prompts(fm, PROMPTS)
    print(f"[Setup] Encoded {len(z_txt_list)} prompts, "
          f"embedding dim = {z_txt_list[0].shape[-1]}")

    # ── 3. Rollout function ───────────────────────────────────────────────────
    rollout_fn = build_rollout_fn(substrate, fm, time_sampling=TIME_SAMPLING)
    print("[Setup] Rollout function built and jitted.")

    # ── 4. Hard bounds from substrate ────────────────────────────────────────
    global_xl, global_xu = get_substrate_bounds(FlattenSubstrateParameters)
    print(f"[Setup] Substrate hard bounds: n_var={len(global_xl)}")

    # ── 5. Load verified extreme solutions ───────────────────────────────────
    print("\n[Setup] Loading extreme solutions ...")
    extremes = load_extremes(EXTREME_PATHS)
    print(f"[Setup] Extremes loaded: shape={extremes.shape}")

    # ── 6. Generate parent pool from optima via Gaussian perturbation ─────────
    parents_list = []
    for i, x_star in enumerate(extremes):
        p = generate_parents_from_optimum(
            x_star=x_star,
            n_parents=N_PARENTS_PER_PROMPT,
            noise_scale=NOISE_SCALE,
            xl=global_xl,
            xu=global_xu,
            rng=np.random.default_rng(BASE_SEED + i),
        )
        parents_list.append(p)
        print(f"[Setup] Generated {len(p)} parents for prompt {i} "
              f"(noise_scale={NOISE_SCALE})")

    parents = np.vstack(parents_list)   # (18, n_var)

    # ── 7. Compute search bounds ──────────────────────────────────────────────
    # Use extremes + parents as seeds so the bounds cover the full
    # initial population distribution
    all_seeds = np.vstack([extremes, parents])
    xl, xu = compute_bounds_from_seeds(
        substrate=substrate,
        seeds=all_seeds,
        dyn_margin=DYN_MARGIN,
        init_margin=INIT_MARGIN,
        global_xl=global_xl,
        global_xu=global_xu,
    )
    print("[Setup] Search bounds computed from extremes + parents.")

    # ── 8. Problem ────────────────────────────────────────────────────────────
    problem = MOO_ASALProblem(
        rollout_fn=rollout_fn,
        z_txt_list=z_txt_list,
        xl=xl,
        xu=xu,
        base_seed=BASE_SEED,
    )
    print(f"[Setup] Problem ready: n_var={problem.n_var}, n_obj={problem.n_obj}")

    # ── 9. Checkpoint / resume ────────────────────────────────────────────────
    ckpt          = load_latest_checkpoint(CHECKPOINT_DIR)
    prior_archive = load_archive(ARCHIVE_PATH)
    start_gen     = ckpt["gen"] if ckpt is not None else 0
    remaining_gen = N_GEN - start_gen

    if remaining_gen <= 0:
        print(f"[Done] All {N_GEN} generations already completed.")
        return

    # ── 10. NSGA-II ───────────────────────────────────────────────────────────
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    # ── 11. Population injection ──────────────────────────────────────────────
    rng = np.random.default_rng(BASE_SEED)

    if ckpt is not None:
        resume_pop = Population.new("X", ckpt["pop_X"], "F", ckpt["pop_F"])
        algorithm.initialization.sampling = resume_pop
        print(f"[Resume] Resuming from generation {start_gen}, "
              f"{remaining_gen} generations remaining.")
    else:
        init_pop = build_initial_population(
            problem=problem,
            extremes=extremes,
            parents=parents,
            n_random=N_RANDOM,
            rng=rng,
        )
        algorithm.initialization.sampling = init_pop

    # ── 12. Callback ──────────────────────────────────────────────────────────
    callback = ArchiveAndCheckpointCallback(
        archive_path=ARCHIVE_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        initial_archive=prior_archive,
    )

    # ── 13. Run ───────────────────────────────────────────────────────────────
    print(f"\n[Run] NSGA-II | pop={POP_SIZE} | "
          f"n_gen={remaining_gen} | n_obj={problem.n_obj}")
    print("=" * 60)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", remaining_gen),
        seed=BASE_SEED,
        callback=callback,
        verbose=False,
    )

    # ── 14. Save results ──────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.save(os.path.join(RESULT_DIR, "pareto_X.npy"), res.X)
    np.save(os.path.join(RESULT_DIR, "pareto_F.npy"), res.F)

    print(f"\n[Done] Optimisation complete.")
    print(f"       Pareto front size : {len(res.F)}")
    print(f"       Total archive size: {len(callback.archive)}")
    print(f"       Results saved to  : {RESULT_DIR}/")


if __name__ == "__main__":
    main()