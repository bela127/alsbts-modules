from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
import numpy as np

from alts.core.blueprint import Blueprint

from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.stopping_criteria import TimeStoppingCriteria
from alts.modules.data_process.observable_filter import NoObservableFilter
from alts.core.experiment_modules import ExperimentModules
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.evaluator import PrintExpTimeEvaluator

from alts.core.data.data_pools import SPRDataPools
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.query.query_sampler import FixedQuerySampler
from alts.core.blueprint import Blueprint
from alts.modules.evaluator import PrintExpTimeEvaluator

from alts.modules.data_process.time_source import IterationTimeSource

from alts.core.oracle.oracles import POracles
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.data_process.process import DelayedStreamProcess
from alts.modules.stopping_criteria import TimeStoppingCriteria


from alts.modules.oracle.data_source import BrownianDriftDataSource

from alts.modules.data_process.observable_filter import NoObservableFilter


from alsbts.modules.estimator import PassThroughEstimator
from alsbts.modules.query.query_selector import StreamQuerySelector

from alts.modules.evaluator import LogOracleEvaluator

from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider

from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alts.modules.evaluator import LogAllEvaluator, LogTVPGTEvaluator

from alsbts.modules.evaluator import EstimatorEvaluator

from alsbts.modules.selection_criteria import FixedIntervalSelectionCriteria

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from alts.core.data_process.time_source import TimeSource
    from alts.core.data_process.process import Process
    from alts.core.stopping_criteria import StoppingCriteria
    from alts.core.data_process.observable_filter import ObservableFilter
    from alts.core.data.queried_data_pool import QueriedDataPool
    from alts.core.query.query_sampler import QuerySampler
    from alts.core.experiment_modules import ExperimentModules
    from alts.core.evaluator import Evaluator
    



stop_time = 1000

@dataclass
class SbBlueprint(Blueprint):
    repeat: int = 1

    time_source: TimeSource = IterationTimeSource(time_step=0.5)#0.05)

    oracles: POracles = POracles(process=FCFSQueryQueue())

    process: Process = DelayedStreamProcess(
        stop_time=stop_time,
        data_source=BrownianDriftDataSource(reinit=True),
    )

    stopping_criteria: StoppingCriteria = TimeStoppingCriteria(stop_time=stop_time)

    data_pools: SPRDataPools = SPRDataPools(
        stream=FlatQueriedDataPool(),
        process=FlatQueriedDataPool(),
        result=FlatQueriedDataPool(),
    )

    initial_query_sampler: QuerySampler = StreamQuerySampler()

    experiment_modules: ExperimentModules = StreamExperiment(
        query_selector=StreamQuerySelector(
            query_optimizer=NoQueryOptimizer(query_sampler=StreamQuerySampler(), selection_criteria=FixedIntervalSelectionCriteria(time_interval=3)),
            query_decider=ThresholdQueryDecider(threshold=0.0),
            ),
        estimator=PassThroughEstimator(),
    )

    evaluators: Iterable[Evaluator] = (PrintExpTimeEvaluator(), LogOracleEvaluator(), LogAllEvaluator(), LogTVPGTEvaluator(), EstimatorEvaluator())

