from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alsbts.modules.behavior import Behavior

if TYPE_CHECKING:
    from typing import Tuple, List, Union
    from typing_extensions import Self
    from nptyping import NDArray, Shape, Number

from alts.core.oracle.data_source import TimeDataSource, DataSource
from alts.core.configuration import Required, is_set, init, pre_init, post_init

@dataclass
class TimeBehaviorDataSource(TimeDataSource):

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


@dataclass
class SeismicTimeDataSource(TimeDataSource):

    query_shape: Tuple[int,...] = init(default=(1,))
    result_shape: Tuple[int,...] = init(default=(1,))
    reinit: bool = init(default=True)
    stop_time: float = pre_init(default=1000)
    current_time: float = pre_init(default=0)
    trace_nr: Union[int, Tuple[int,...]] = init(default=(0,1,2,3,4,5,6,7,8,9,10,11))
    change_times: NDArray[Shape["change_times"], Number] = pre_init(default=None)
    change_values: NDArray[Shape["change_values"], Number] = pre_init(default=None)

    def post_init(self):
        super().post_init()
        import os
        from obspy import read

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)

        stream = read(f'{dir_path}/ContinuousActive-SourceSeismicMonitoring/1903.dat')

        self.init_singleton(stream)

    def init_singleton(self, stream):
        start_index = 100
        stop_index = 1500
    
        if self.change_values is None or self.reinit == True:
            from random import choice
            import pandas as pd
            
            if isinstance(self.trace_nr, tuple):
                self.trace_nr = choice(self.trace_nr)
            
            trace = stream.traces[self.trace_nr]
            data = trace.data
            df = pd.Series(data).rolling(window=100).mean()
            values = np.asarray(df[start_index:stop_index])
            values = values - 1000
            values = values / 200
            self.change_values = values
            self.change_times = np.linspace(0, self.stop_time, stop_index-start_index)


    @property
    def exhausted(self):
        return self.current_time < self.stop_time

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:
        times = queries
        self.current_time = times[-1,0]

        indices = np.searchsorted(self.change_times, times[...,0],side='right') -1

        results = self.change_values[indices][:,None]

        return queries, results

    def __call__(self, **kwargs) -> Self:
        obj: SeismicTimeDataSource = super().__call__( **kwargs)
        obj.change_values = self.change_values
        obj.change_times = self.change_times
        obj.trace_nr = self.trace_nr
        return obj


@dataclass
class SeismicDataSource(DataSource):

    query_shape: Tuple[int,...] = init(default=(2,))
    result_shape: Tuple[int,...] = init(default=(1,))
    reinit: bool = init(default=True)
    stop_time: float = pre_init(default=1000)
    current_time: float = pre_init(default=0)
    trace_nr: Union[int, Tuple[int,...]] = init(default=(0,1,2,3,4,5,6,7,8,9,10,11))
    change_times: NDArray[Shape["change_times"], Number] = pre_init(default=None)
    change_values: NDArray[Shape["change_values"], Number] = pre_init(default=None)

    def post_init(self):
        super().post_init()
        import os
        from obspy import read

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)

        stream = read(f'{dir_path}/ContinuousActive-SourceSeismicMonitoring/1903.dat')

        self.init_singleton(stream)

    
    def init_singleton(self, stream):
        start_index = 100
        stop_index = 1500
    
        if self.change_values is None or self.reinit == True:
            from random import choice
            import pandas as pd
            
            if isinstance(self.trace_nr, tuple):
                self.trace_nr = choice(self.trace_nr)
            
            trace = stream.traces[self.trace_nr]
            data = trace.data
            df = pd.Series(data).rolling(window=100).mean()
            values = np.asarray(df[start_index:stop_index])
            values = values - 1000
            values = values / 20
            self.change_values = values
            self.change_times = np.linspace(0, self.stop_time, stop_index-start_index)

    @property
    def exhausted(self):
        return self.current_time < self.stop_time

    def query(self, queries: NDArray[ Shape["query_nr, ... query_dim"], Number]) -> Tuple[NDArray[Shape["query_nr, ... query_dim"], Number], NDArray[Shape["query_nr, ... result_dim"], Number]]:
        times = queries[:,:1]
        self.current_time = times[-1,0]

        indices = np.searchsorted(self.change_times, times[...,0],side='right') -1

        results = self.change_values[indices][:,None]

        return queries, results
    
    def __call__(self, **kwargs) -> Self:
        obj: SeismicDataSource = super().__call__( **kwargs)
        obj.change_values = self.change_values
        obj.change_times = self.change_times
        obj.trace_nr = self.trace_nr
        return obj