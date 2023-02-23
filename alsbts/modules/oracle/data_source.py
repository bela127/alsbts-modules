from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alsbts.modules.behavior import Behavior

if TYPE_CHECKING:
    from typing import Tuple, List
    from nptyping import NDArray, Shape, Number

from alts.core.oracle.data_source import DataSource
from alts.core.configuration import Required, is_set

@dataclass
class TimeBehaviorDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    behavior: Required[Behavior] = None
    change_times: NDArray[Shape["change_times"], Number] = field(init=False)
    change_values: NDArray[Shape["change_values"], Number] = field(init=False)
    current_time: float = field(init=False, default=0)

    def __post_init__(self):
        self.behavior = is_set(self.behavior)()
        self.change_times, self.change_values = self.behavior.behavior()


    @property
    def exhausted(self):
        return self.current_time < self.behavior.stop_time

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:
        times = queries
        self.current_time = times[-1,0]

        indices = np.searchsorted(self.change_times, times[...,0],side='right') -1

        results = self.change_values[indices][:,None]

        return queries, results
        