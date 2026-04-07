import os
import util
from pymoo.core.callback import Callback

from algorithms.extractors import extract_population, extract_front
from runtime.archive import build_generation_records, update_novelty_archive
from runtime.logging import format_moo_log
from runtime.checkpoint import make_checkpoint_state, save_checkpoint


class ArchiveAndCheckpointCallback(Callback):
    def __init__(
        self,
        save_dir,
        checkpoint_dir,
        save_every=100,
        start_gen=0,
        initial_archive_size=0,
        novelty_archive=None,
        algo_name="NSGA-II",
        front_mode="opt",
        args=None,
        problem=None,
        algo_meta=None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.start_gen = start_gen
        self.archive_size = initial_archive_size
        self.novelty_archive = novelty_archive
        self.algo_name = algo_name
        self.front_mode = front_mode
        self.args = args
        self.problem = problem
        self.algo_meta = algo_meta

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def notify(self, algorithm):
        local_gen = algorithm.n_gen
        abs_gen = self.start_gen + local_gen

        _, X_all, F_all, _ = extract_population(algorithm)
        _, X_front, F_front, BD_front = extract_front(algorithm, mode=self.front_mode)

        new_records = build_generation_records(abs_gen, X_front, F_front)
        update_novelty_archive(self.novelty_archive, BD_front)

        if self.save_dir is not None:
            util.append_pkl(self.save_dir, "archive", new_records)

        self.archive_size += len(new_records)

        if abs_gen % self.save_every == 0 and self.checkpoint_dir is not None:
            ckpt = make_checkpoint_state(
                abs_gen=abs_gen,
                X_all=X_all,
                F_all=F_all,
                args=self.args,
                problem=self.problem,
                algo_meta=self.algo_meta,
            )
            save_checkpoint(self.checkpoint_dir, f"ckpt_gen{abs_gen:04d}", ckpt)
            save_checkpoint(self.checkpoint_dir, "latest", ckpt)

        msg = format_moo_log(
            problem=self.problem,
            algo_name=self.algo_name,
            gen=abs_gen,
            front_size=len(F_front),
            archive_size=self.archive_size,
            F_front=F_front,
        )
        print(msg, flush=True)