from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alts.core.data.data_pool import DataPool
from alts.core.oracle.data_source import DataSource
from alts.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List, Any

    from nptyping import  NDArray, Number, Shape

    from typing_extensions import Self

@dataclass
class TimeDependentDataSource(DataSource):
    query_shape: Tuple[int,...] = None
    result_shape: Tuple[int,...] = None

    data_source: DataSource = None
    end_time: float = 900
    time_steps: float = 0.2
    time_slice: Tuple[int,...] = (0,)

    last_time = 0

    def __post_init__(self):
        self.data_source = self.data_source()
        self.query_shape: Tuple[int,...] = self.data_source.query_shape
        self.result_shape: Tuple[int,...] = self.data_source.result_shape

    def query(self, queries):
        number = queries.shape[0]
        times = np.linspace(self.last_time, number*self.time_steps, number)
        self.last_time = number*self.time_steps

        queries[self.time_slice] = times

        queries, results = self.data_source.query(queries)
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        return self.data_source.query_pool
    
    @property
    def exhausted(self):
        return self.last_time >= self.end_time

