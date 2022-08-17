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
    result_shape: Tuple[int,...] = (9,)

    exp_setup: ExperimentSetup = None

    def query(self, queries):
        estimated_vs, measure = queries[0]

        if measure == 1:
            self.exp_setup.trigger_vs_measurement()
        else:
            self.exp_setup.continue_sim()

        measurement_in_progress, measured_vs = self.exp_setup.last_vs_measurement()
        
        timeOutput , voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput= self.exp_setup.observable_grid_quantities()


        vs_gt = self.exp_setup.gt_vs()
        rvs = self.exp_setup.estimate_rvs(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput)
        change_point = self.exp_setup.rvs_change(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs)


        result = np.asarray((timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs, change_point, measurement_in_progress, measured_vs))
        results = np.asarray([result])

        print("VS_GT: ",vs_gt)
        return queries, results

    @property
    def exhausted(self):
        return not self.exp_setup.is_running

    
    @property
    def query_pool(self) -> QueryPool:
        query_ranges = np.asarray(((np.nan, np.nan), (0, 1)))
        query_pool = QueryPool(query_count=2, query_shape=self.query_shape, query_ranges=query_ranges)
        query_pool.add_queries(np.asarray([[np.nan, 0],[np.nan, 1]]))
        return query_pool

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

    def __call__(self, **kwargs) -> Self:
        obj = super().__call__(**kwargs)
        obj.exp_setup = obj.exp_setup()
        return obj

    
