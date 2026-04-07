def summarize_objectives(F_front):
    vals = -F_front
    best_vals = vals.max(axis=0)
    mean_vals = vals.mean(axis=0)
    return best_vals, mean_vals


def format_objective_summary(problem, F_front):
    best_vals, mean_vals = summarize_objectives(F_front)

    obj_names = getattr(problem, "objective_names", [f"obj_{i}" for i in range(F_front.shape[1])])
    obj_roles = getattr(problem, "objective_roles", ["objective"] * F_front.shape[1])

    parts = []
    for name, role, best, mean in zip(obj_names, obj_roles, best_vals, mean_vals):
        parts.append(f"{name}({role}): best={best:.4f}, mean={mean:.4f}")
    return " | ".join(parts)


def format_moo_log(problem, algo_name, gen, front_size, archive_size, F_front):
    summary = format_objective_summary(problem, F_front)
    return (
        f"[{algo_name} | Gen {gen:04d}] "
        f"front={front_size} | "
        f"archive={archive_size} | "
        f"{summary}"
    )