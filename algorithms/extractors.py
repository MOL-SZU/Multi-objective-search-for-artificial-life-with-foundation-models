import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def extract_population(algorithm):
    pop = algorithm.pop
    X = pop.get("X")
    F = pop.get("F")
    BD = pop.get("BD") if pop.has("BD") else None
    return pop, X, F, BD


def extract_front(algorithm, mode="opt"):
    if mode == "opt":
        front = algorithm.opt

    elif mode == "nondominated_from_pop":
        pop = algorithm.pop
        F = pop.get("F")
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        front = pop[nd_idx]

    elif mode == "population":
        front = algorithm.pop

    else:
        raise ValueError(f"Unknown front extraction mode: {mode}")

    X = front.get("X")
    F = front.get("F")
    BD = front.get("BD") if front.has("BD") else None
    return front, X, F, BD