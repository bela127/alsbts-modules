from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from alts.core.data.data_pool import DataPool
from alts.core.oracle.data_source import DataSource
from alts.core.query.query_pool import QueryPool

from alvsts.modules.experiment_setup import ExperimentSetup

if TYPE_CHECKING:
    from typing import Tuple, List, Any

    from nptyping import  NDArray, Number, Shape

    from typing_extensions import Self



@dataclass
class VSSimulationDataSource(DataSource):

    query_shape: Tuple[int,...] = (2,)
    result_shape: Tuple[int,...] = (8,)

    exp_setup: ExperimentSetup = None

    def query(self, queries):
        print("query:", queries)
        estimated_vs, measure = queries[0]

        self.exp_setup.simulation_step()
        
        voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, timeOutput = self.exp_setup.observable_grid_quantities()
        if measure == 1:
            vs = self.exp_setup.measure_vs()
        else:
            vs = np.nan

        
        rvs = self.exp_setup.estimate_rvs(timeOutput)
        change_point = self.exp_setup.rvs_change(timeOutput)


        result = np.asarray((timeOutput[-1][0], voltageOutput[-1][0], knewVOutput[-1][0], activePowerOutput[-1][0], reactivePowerOutput[-1][0], rvs, change_point, vs[-1][0]))
        print("result:", result)

        return queries, result

    @property
    def exhausted(self):
        return not self.exp_setup.is_running

    @property
    def query_pool(self) -> QueryPool:
        x_min = 1
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

    
