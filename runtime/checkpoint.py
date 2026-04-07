import os
import pickle
import util


def safe_load_pkl(dir_path, name, default=None, warn=True):
    if dir_path is None:
        return default
    path = os.path.join(dir_path, f"{name}.pkl")
    if not os.path.exists(path):
        return default
    try:
        return util.load_pkl(dir_path, name)
    except (EOFError, pickle.UnpicklingError) as e:
        if warn:
            print(f"[Warn] Failed to load {path}: {type(e).__name__}: {e}")
        return default


def get_archive_len(save_dir):
    archive = safe_load_pkl(save_dir, "archive", default=None, warn=True)
    if archive is None:
        return 0
    if isinstance(archive, list):
        return len(archive)
    print(f"[Warn] archive.pkl is not a list, archive length reset to 0.")
    return 0


def save_checkpoint(checkpoint_dir, name, ckpt):
    if checkpoint_dir is None:
        return
    util.save_pkl(checkpoint_dir, name, ckpt)


def make_checkpoint_state(abs_gen, X_all, F_all, args, problem, algo_meta):
    return {
        "version": 2,
        "resume_mode": "warm",
        "gen": abs_gen,
        "pop_X": X_all.copy(),
        "pop_F": F_all.copy(),
        "algorithm_name": algo_meta["name"],
        "n_var": problem.n_var,
        "n_obj": problem.n_obj,
        "objective_names": getattr(problem, "objective_names", None),
        "objective_roles": getattr(problem, "objective_roles", None),
        "args": vars(args).copy(),
    }


def validate_checkpoint(ckpt, problem, algo_meta):
    required_keys = {"gen", "pop_X", "pop_F"}
    if not isinstance(ckpt, dict) or not required_keys.issubset(ckpt.keys()):
        raise ValueError("Checkpoint is missing required fields.")

    if "n_var" in ckpt and ckpt["n_var"] != problem.n_var:
        raise ValueError(
            f"Checkpoint n_var ({ckpt['n_var']}) does not match current problem.n_var ({problem.n_var})."
        )

    if "n_obj" in ckpt and ckpt["n_obj"] != problem.n_obj:
        raise ValueError(
            f"Checkpoint n_obj ({ckpt['n_obj']}) does not match current problem.n_obj ({problem.n_obj})."
        )

    ckpt_pop_size = len(ckpt["pop_X"])
    if ckpt_pop_size != algo_meta["effective_pop_size"]:
        raise ValueError(
            f"Checkpoint population size ({ckpt_pop_size}) does not match "
            f"the effective population size required by {algo_meta['name']} "
            f"({algo_meta['effective_pop_size']})."
        )


def load_latest_checkpoint(checkpoint_dir):
    if checkpoint_dir is None:
        return None

    latest_path = os.path.join(checkpoint_dir, "latest.pkl")
    if not os.path.exists(latest_path):
        print(f"[Resume] No checkpoint found at {latest_path}")
        return None

    ckpt = safe_load_pkl(checkpoint_dir, "latest", default=None, warn=True)
    if ckpt is None:
        print(f"[Resume] latest.pkl exists but could not be loaded.")
        return None

    required_keys = {"gen", "pop_X", "pop_F"}
    if not isinstance(ckpt, dict) or not required_keys.issubset(ckpt.keys()):
        print(f"[Resume] latest.pkl is invalid, ignoring it.")
        return None

    print(f"[Resume] Checkpoint at gen {ckpt['gen']} ({len(ckpt['pop_X'])} individuals)")
    return ckpt