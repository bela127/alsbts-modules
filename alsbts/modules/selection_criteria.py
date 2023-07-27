from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from alts.core.data.constrains import QueryConstrain

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

    def query_constrain(self) -> QueryConstrain:
        qconst = self.exp_modules.oracles.process.query_constrain()
        queries = self.exp_modules.data_pools.results.queries
        queries = queries.copy()
        vs_estimate = self.exp_modules.estimator.estimate()
        queries[:,0] = vs_estimate
        return QueryConstrain(count=queries.shape[0], shape=qconst.shape, ranges=queries)


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

        queries, est_var = self.exp_modules.estimator.query(queries)
        time = self.exp_modules.data_pools.stream.last_results[0,0]

        scores = est_var[:,1:] - self.std_threshold


        if time <= self.stop_train_time and time >= self.last_query_time + self.time_interval:
            return scores
        else:
            return np.zeros_like(scores)




@dataclass
class ChangeSelectionCriteria(EstimatorSelectionCriteria):
    change_detector: ChangeDetector = init(default_factory=OptimalChangeDetector)
    def post_init(self):
        super().post_init()
        self.change_detector = self.change_detector(exp_modules=self.exp_modules)

    def query(self, queries):
        change = self.change_detector.detect(queries[...,1:])
        return queries, change



