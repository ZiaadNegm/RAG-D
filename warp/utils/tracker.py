import time
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
from dataclasses import dataclass

class ExecutionTrackerIteration:
    def __init__(self, tracker):
        self._tracker = tracker

    def __enter__(self):
        self._tracker.next_iteration()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._tracker.end_iteration()

class Logging_Mode(Enum):
    "Ranging from most detailed to least detailed. Determines what 'special' metrics are logged"
    A = "A"
    B = "B"
    C = "C"

@dataclass
class MetricsA:
        query_id = None
        dataset = None
        query_length = None
        n_clusters_selected = None
        unique_docs = None
        total_token_scores = None
        difficulty = None

@dataclass
class MetricsB:
    MetricsA
    doc_id = None
    score_full = None
    score_centroid_only = None
    num_influential_tokens = None

@dataclass
class MetricsC:
    MetricsB
    token_position = None
    cluster_id = None
    centroid_contribution = None
    residual_contribution = None
    is_max_for_doc = None

A_Logging_Metrics = {}
class SpecialMetrics:
    # TODO: For n_clusters_selected, we should change this into a percentage
    #  of how many centroids vs what was supposed to be touched
    def __init__(self, mode: Logging_Mode):
        self.mode = mode
        self._current = {}
        self._all_iterations = []

    def record(self, name, value):
        self._current[name] = value

    def next_iteration(self):
        self._current = {}
    
    def end_iteration(self):
        self._all_iterations.append(self._current.copy())

    def summary(self):
        print(self.end_iteration)

class ExecutionTracker:
    def __init__(self, name, steps, mode=Logging_Mode.A):
        self._name = name
        self._steps = steps
        self._num_iterations = 0
        self._time = None
        self._time_per_step = {}
        self.specialMetrics = SpecialMetrics(mode)
        for step in steps:
            self._time_per_step[step] = 0
        self._iter_begin = None
        self._iter_time = 0

    def next_iteration(self):
        self._num_iterations += 1
        self._iterating = True
        self._current_steps = []
        self._iter_begin = time.time()
        self.specialMetrics.next_iteration()

    def end_iteration(self):
        tok = time.time()
        if self._steps != self._current_steps:
            print(self._steps, self._current_steps)
        assert self._steps == self._current_steps
        self._iterating = False
        self._iter_time += tok - self._iter_begin
        self.specialMetrics.end_iteration()
        # TODO: Save metrics_per_query: Write to file? -> Clean the set entries

    def iteration(self):
        return ExecutionTrackerIteration(self)

    def begin(self, name):
        assert self._time is None and self._iterating
        self._current_steps.append(name)
        self._time = time.time()

    def end(self, name):
        tok = time.time()
        assert self._current_steps[-1] == name
        self._time_per_step[name] += tok - self._time
        self._time = None

    # TODO: Lets add a finalize so we know when to write away the Set of "Metrics overall"

    def summary(self, steps=None):
        if steps is None:
            steps = self._steps
        iteration_time = self._iter_time / self._num_iterations
        breakdown = [
            (step, self._time_per_step[step] / self._num_iterations) for step in steps
        ]
        self.specialMetrics.summary()
        return iteration_time, breakdown

    def record(self, name, value):
        self.specialMetrics.record(name, value)

    def as_dict(self):
        return {
            "name": self._name,
            "steps": self._steps,
            "time_per_step": self._time_per_step,
            "num_iterations": self._num_iterations,
            "iteration_time": self._iter_time,
        }

    @staticmethod
    def from_dict(data):
        tracker = ExecutionTracker(data["name"], data["steps"])
        tracker._time_per_step = data["time_per_step"]
        tracker._num_iterations = data["num_iterations"]
        tracker._iter_time = data["iteration_time"]
        return tracker

    def __getitem__(self, key):
        assert key in self._steps
        return self._time_per_step[key] / self._num_iterations

    def display(self, steps=None, bound=None):
        iteration_time, breakdown = self.summary(steps)
        df = pd.DataFrame(
            {
                "Task": [x[0] for x in breakdown],
                "Duration": [x[1] * 1000 for x in breakdown],
            }
        )
        df["Start"] = df["Duration"].cumsum().shift(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 2))

        for i, task in enumerate(df["Task"]):
            start = df["Start"][i]
            duration = df["Duration"][i]
            ax.barh("Tasks", duration, left=start, height=0.5, label=task)

        plt.xlabel("Latency (ms)")
        accumulated = round(sum([x[1] for x in breakdown]) * 1000, 1)
        actual = round(iteration_time * 1000, 1)
        plt.title(
            f"{self._name} (iterations={self._num_iterations}, accumulated={accumulated}ms, actual={actual}ms)"
        )
        ax.set_yticks([])
        ax.set_ylabel("")

        if bound is not None:
            ax.set_xlim([0, bound])

        plt.legend()
        plt.show()


class NOPTracker:
    def __init__(self):
        pass

    def next_iteration(self):
        pass

    def begin(self, name):
        pass  # NOP

    def end(self, name):
        pass  # NOP

    def end_iteration(self):
        pass

    def summary(self):
        raise AssertionError

    def record(self, name, value):
        pass  # NOP

    def display(self):
        raise AssertionError
