#!/usr/bin/env python3
"""
The module docstring
"""

import os
from typing import List, Tuple, Type, TypeVar
import numpy as np

T = TypeVar("T", bound="VinaLog")


class VinaLog:
    """
    This represents a output log from a VINA docking run
    """

    __slots__ = ["_raw", "_data", "_seed"]

    def __init__(self: Type[T], raw: List[str], data: np.ndarray, seed: int):
        self._raw: List[str] = raw.copy()
        self._data: np.ndarray = data
        self._seed = seed

    @classmethod
    def from_file(cls: Type[T], filename: str) -> T:
        """
        Create a VinaLog object from file
        """

        raw: List[str] = []
        with open(filename, "r") as file:
            raw = file.readlines()

        data_sentinal: str = "-----+------------+----------+----------"
        seed_sentinal: str = "Using random seed"

        data_loading: bool = False
        data: List[float] = []
        data_rows: List[Tuple(float, float, float)] = []

        seed = False

        for line in map(lambda l: l.split("\n")[0], raw):

            if seed_sentinal in line:
                seed = int(line.split()[-1])
                continue

            if data_sentinal in line:
                data_loading = True
                continue

            if data_loading:
                tokens = line.split()

                try:
                    int(tokens[0])

                except ValueError:
                    data_loading = False
                    continue

                data = tuple(map(float, tokens[1:]))

                data_rows.append(data)

        if not seed:
            raise Exception("Could not parse {:s}".format(filename))

        dtype = [
            (("affinity (kcal/mol)", "score"), "f8"),
            (("dist from best mode lower bound", "rmsd_l"), "f8"),
            (("dist from best mode upper bound", "rmsd_u"), "f8"),
        ]

        data = np.array(data_rows, dtype=dtype)

        return cls(raw, data, seed)

    def to_file(self: Type[T], filename: str) -> None:
        """
        Save this log to file
        """
        with open(filename, "w") as file:
            for line in self._raw:
                file.write(line)

    def filter_greater_than(self: Type[T], threshhold: float) -> int:
        s = self.scores < threshhold
        n = self.n_scores
        self._data = self._data[s]
        return self.n_scores - n

    @property
    def scores(self: Type[T]) -> np.array:
        """
        The docking score
        """
        return self._data["score"]

    @property
    def rmsd_l(self: Type[T]) -> np.array:
        """
        The RMSD lower limit
        """
        return self._data["rmsd_l"]

    @property
    def rmsd_u(self: Type[T]) -> np.ndarray:
        """
        The RMSD upper limit
        """
        return self._data["rmsd_u"]

    @property
    def data(self: Type[T]) -> np.ndarray:
        """
        The data of the log file
        """
        return self._data

    @property
    def seed(self: Type[T]) -> int:
        """
        The seed used in the docking run
        """
        return self._seed

    @property
    def n_scores(self: Type[T]) -> int:
        """
        The number of score this log contains
        """
        return self._data.shape[0]


class VinaDockCollector:
    def __init__(self, base_dirs=["."], select=None):
        self._base_dirs = base_dirs
        self._select = select
        self._logs = {}

    def collect(self, verbose=False, raise_on_error=False, threshhold=np.inf):

        for base_dir in self._base_dirs:
            self._logs[base_dir] = {}
            print("Collecting for", base_dir)
            for struct in os.walk(base_dir):
                if verbose:
                    print("Parsing", struct[0])
                if "slurm" in struct[0]:
                    continue
                for file in struct[2]:
                    # print("Consider file", file, "...", end=" ")
                    tokens = file.split(".")
                    suffix = tokens[-1]
                    if suffix != "log":
                        # print("skip since suffix=", suffix)
                        continue
                    # print("Checking if select=", self._select,
                    # "matches", tokens[:-1])
                    match = False
                    for select in self._select:
                        match = match or any([select == t for t in tokens[:-1]])

                    if match:
                        if verbose:
                            print("Loading", file)
                        key = os.path.join(struct[0], file)
                        try:
                            log = VinaLog.from_file(key)
                            filtered = log.filter_greater_than(threshhold)
                            if verbose:
                                print("Filtered", filtered, "results")
                        except Exception as e:
                            if raise_on_error:
                                raise
                            else:
                                print(e, ": skipping")

                        if log.n_scores > 0:
                            self._logs[base_dir][key] = log

    def analyze(self):

        for base in self._base_dirs:
            print("Analyzing", base)
            scores = {k: log.scores.min() for k, log in self._logs[base].items()}

            best_score = 0.0
            best_name = None
            for k, v in scores.items():
                if best_name is None or v < best_score:
                    best_name = k
                    best_score = v

            vals = np.array(list(scores.values()))

            n_vals = vals.shape[0]

            print("###################################")
            print("Results for", base)

            print("Number of results:", n_vals)
            if n_vals == 0:
                return
            print("Best score:", best_score)
            print("Best frame:", best_name)
            print("Average best score:", vals.mean())
            print("Stddev best score:", vals.std())

    def hist(self, min_only=False, savefig=False):
        import matplotlib.pyplot as plt

        plotted = False
        for i, base in enumerate(self._logs):
            values = self._logs[base].values()
            if min_only:
                scores = np.array([log.scores.min() for log in values])

            else:
                scores = np.concatenate([log.scores for log in values])

            if len(scores) == 0:
                continue
            # plt.hist(scores, bins=8, lw=4, histtype="step", density=True, label=base)
            name = base.split('-')
            if len(name) > 1:
                name = name[1]
            plt.boxplot(scores, positions=[i], labels=[name], widths=.8)
            plotted = True

        if plotted:
            # plt.legend()
            title_str = "Docking of ligands to site {:s}"
            title = title_str.format(", ".join(self._select))
            plt.suptitle(title)
            if savefig:
                plt.savefig("-".join(self._select) + ".png")
            plt.ylabel("Affinity (kcal/mol)")
            plt.show()
