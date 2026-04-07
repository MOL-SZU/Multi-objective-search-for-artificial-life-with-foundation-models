def build_generation_records(gen, X, F):
    return [{"gen": gen, "X": x.copy(), "F": f.copy()} for x, f in zip(X, F)]


def update_novelty_archive(novelty_archive, BD):
    if novelty_archive is None or BD is None:
        return
    novelty_archive.extend(BD)