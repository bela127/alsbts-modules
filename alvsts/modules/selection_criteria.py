from __future__ import annotations
import imp
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from alts.core.data.data_pool import DataPool

from alts.core.query.selection_criteria import SelectionCriteria, NoSelectionCriteria
#from aldtts.modules.experiment_modules import DependencyExperiment, InterventionDependencyExperiment


if TYPE_CHECKING:
    from typing import Optional
    from typing_extensions import Self #type: ignore
    #from aldtts.modules.test_interpolation import TestInterpolator
    #from aldtts.modules.dependency_test import DependencyTest
    from alts.core.oracle.data_source import DataSource


@dataclass
class TestSelectionCriteria(SelectionCriteria):
    test_interpolator: Optional[TestInterpolator] = field(init=False, default=None)
    
    @property
    def query_pool(self):
        return self.test_interpolator.query_pool

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)

        if isinstance(exp_modules, InterventionDependencyExperiment):
            obj.test_interpolator = exp_modules.test_interpolator
        else:
            raise ValueError()

        return obj

@dataclass
class UKFUncertaintySelectionCriteria(TestSelectionCriteria):
    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = 1 - mean_p

        scores = np.repeat(score,2)

        return scores

@dataclass
class RVSChangeSelectionCriteria(TestSelectionCriteria):
    def query(self, queries):

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))

        t, p, u = self.test_interpolator.query(test_queries)

        mean_p = np.mean(p, axis=1)

        score = 1 - mean_p

        scores = np.repeat(score,2)

        return scores

@dataclass
class FixedIntervalSelectionCriteria(SelectionCriteria):
    time_interval: float = 10.0

    last_query_time: float = field(init=False, default=0)

    def query(self, queries):

        time = self.exp_modules.queried_data_pool.last_results[0,0]

        if self.last_query_time <= time:
            scores = np.asarray([0, 1])
            self.last_query_time = time
        else:
            scores = np.asarray([1, 0])

        return scores
