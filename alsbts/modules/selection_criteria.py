from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from alts.core.query.query_pool import QueryPool

from alts.core.configuration import init, pre_init, post_init
from alts.core.query.selection_criteria import SelectionCriteria
from alsbts.core.experiment_modules import StreamExperiment

from alsbts.core.change_detector import ChangeDetector
from alsbts.modules.change_detector import OptimalChangeDetector

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore

    from alsbts.core.estimator import Estimator


@dataclass
class EstimatorSelectionCriteria(SelectionCriteria):
    exp_modules: StreamExperiment = post_init()

    def __post_init__(self):
        super().__post_init__()

    @property
    def query_pool(self) -> QueryPool:
        query_pool = self.exp_modules.oracle_data_pool.copy()
        
        def all_queries():
            queries = self.exp_modules.oracle_data_pool.all_queries()
            vs_estimate = self.estimator.estimate()
            queries = queries.copy()
            queries[:,0] = vs_estimate
            return queries
        
        query_pool.all_queries = all_queries
        return query_pool


@dataclass
class FixedIntervalSelectionCriteria(SelectionCriteria):
    time_interval: float = init(default=10.0)

    last_query_time: float = pre_init(default=0)

    def query(self, queries):

        times = queries[:,0]

        lq_times = (times - self.last_query_time) % self.time_interval

        mask = lq_times[:-1] >= lq_times[1:]
        mask = np.concatenate((np.asarray([times[0] - self.last_query_time >= self.time_interval]),mask))
        
        scores = np.zeros((times.shape[0],1))

        scores[mask] = 1

        time = times[mask]

        if time.shape[0] > 0:
            self.last_query_time = time[-1]

        return queries, scores

@dataclass
class PreTrainIntervalSelectionCriteria(SelectionCriteria):
    time_interval: float = 2.0
    stop_train_time: float = 200

    last_query_time: float = pre_init(default=0)

    def query(self, queries):

        times = queries[:, :1]

        scores = times - (self.last_query_time + self.time_interval - 1)
        scores[times > self.stop_train_time] = 0
        measure_times = times[scores>0]
        if measure_times:
            self.last_query_time = measure_times[-1]

        return queries, scores

@dataclass
class STDSelectionCriteria(EstimatorSelectionCriteria):
    std_threshold: float = 0.005

    def query(self, queries):

        queries, est_var = self.exp_modules.estimator.query(queries)

        scores = est_var[:,1:] - self.std_threshold

        return queries, scores


@dataclass
class STDPreTrainSelectionCriteria(EstimatorSelectionCriteria):
    std_threshold: float = 1.0
    time_interval: float = 2.5
    stop_train_time: float = 50

    last_query_time: float = pre_init(default=0)

    def query(self, queries):

        std = self.estimator.last_std
        time = self.exp_modules.queried_data_pool.last_results[0,0]


        if std > self.std_threshold or time <= self.stop_train_time and time >= self.last_query_time + self.time_interval:
            scores = np.asarray([0, 1]) #Do measure
            self.last_query_time = time
        else:
            scores = np.asarray([1, 0]) #Do not measure

        return scores


@dataclass
class ChangeSelectionCriteria(EstimatorSelectionCriteria):
    change_detector: ChangeDetector = init(default_factory=OptimalChangeDetector)
    def __post_init__(self):
        super().__post_init__()
        self.change_detector = self.change_detector()

    def query(self, queries):
        change = self.change_detector.detect(queries[...,1:])
        return queries, change



