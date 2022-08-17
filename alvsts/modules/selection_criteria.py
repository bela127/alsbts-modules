from __future__ import annotations
import imp
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from alts.core.query.query_pool import QueryPool

from alts.core.query.selection_criteria import SelectionCriteria, NoSelectionCriteria
from alsbts.core.experiment_modules import StreamExperiment

if TYPE_CHECKING:
    from typing import Optional
    from typing_extensions import Self #type: ignore
    from alts.core.oracle.data_source import DataSource

    from alsbts.core.estimator import Estimator


@dataclass
class EstimatorSelectionCriteria(SelectionCriteria):
    estimator: Estimator = field(init=False)

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)

        if isinstance(exp_modules, StreamExperiment):
            obj.estimator = exp_modules.estimator
        else:
            raise ValueError()

        return obj

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
class UKFUncertaintySelectionCriteria(EstimatorSelectionCriteria):
    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = 1 - mean_p

        scores = np.repeat(score,2)

        return scores

@dataclass
class RVSChangeSelectionCriteria(EstimatorSelectionCriteria):
    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = 1 - mean_p

        scores = np.repeat(score,2)

        return scores

@dataclass
class FixedIntervalSelectionCriteria(EstimatorSelectionCriteria):
    time_interval: float = 10.0

    last_query_time: float = field(init=False, default=0)

    def query(self, queries):

        time = self.exp_modules.queried_data_pool.last_results[0,0]

        if time >= self.last_query_time + self.time_interval:
            scores = np.asarray([0, 1]) #Do measure
            self.last_query_time = time
        else:
            scores = np.asarray([1, 0]) #Do not measure

        return scores

@dataclass
class ChangeSelectionCriteria(EstimatorSelectionCriteria):

    def query(self, queries):

        change = self.exp_modules.queried_data_pool.last_results[0,6]

        if change > 0:
            scores = np.asarray([0, 1]) #Do measure
        else:
            scores = np.asarray([1, 0]) #Do not measure

        return scores



