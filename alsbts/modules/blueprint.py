from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
import numpy as np

from alts.core.blueprint import Blueprint

from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.stopping_criteria import TimeStoppingCriteria
from alts.modules.data_process.observable_filter import NoObservableFilter
from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.core.experiment_modules import ExperimentModules
from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.evaluator import PrintExpTimeEvaluator

from alts.modules.queried_data_pool import FlatQueriedDataPool
from alts.modules.query.query_sampler import FixedQuerySampler
from alts.core.blueprint import Blueprint
from alts.modules.evaluator import PrintExpTimeEvaluator

from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.data_process.time_behavior import DataSourceTimeBehavior
from alts.modules.oracle.query_queue import FCFSQueryQueue
from alts.modules.data_process.process import DataSourceProcess
from alts.modules.stopping_criteria import TimeStoppingCriteria


from alts.modules.oracle.data_source import BrownianDriftDataSource

from alts.modules.data_process.observable_filter import NoObservableFilter


from alsbts.modules.estimator import PassThroughEstimator
from alsbts.modules.query.query_selector import StreamQuerySelector

from alts.modules.evaluator import LogOracleEvaluator

from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider

from alsbts.modules.oracle.data_source import TimeBehaviorDataSource
from alsbts.modules.behavior import RandomTimeUniformBehavior

from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alts.modules.evaluator import LogAllEvaluator, LogTVPGTEvaluator

from alsbts.modules.evaluator import EstimatorEvaluator

from alts.modules.query.selection_criteria import AllSelectionCriteria

if TYPE_CHECKING:
    from typing import Iterable, Optional

    from alts.core.data_process.time_source import TimeSource
    from alts.core.data_process.time_behavior import TimeBehavior
    from alts.core.data_process.process import Process
    from alts.core.stopping_criteria import StoppingCriteria
    from alts.core.data_process.observable_filter import ObservableFilter
    from alts.core.data.queried_data_pool import QueriedDataPool
    from alts.core.query.query_sampler import QuerySampler
    from alts.core.experiment_modules import ExperimentModules
    from alts.core.evaluator import Evaluator
    from alts.core.oracle.query_queue import QueryQueue



stop_time = 1000

@dataclass
class SbBlueprint(Blueprint):
    repeat: int = 1

    time_source: TimeSource = IterationTimeSource()
    time_behavior: TimeBehavior = DataSourceTimeBehavior(
        data_source= TimeBehaviorDataSource(behavior=RandomTimeUniformBehavior(stop_time=stop_time))
    )
    query_queue: QueryQueue = FCFSQueryQueue()

    process: Process = DataSourceProcess(
        data_source=BrownianDriftDataSource(reinit=True),
    )

    stopping_criteria: StoppingCriteria = TimeStoppingCriteria(stop_time=stop_time)

    stream_data_pool: QueriedDataPool = FlatQueriedDataPool()
    process_data_pool: QueriedDataPool = FlatQueriedDataPool()
    result_data_pool: QueriedDataPool = FlatQueriedDataPool()

    initial_query_sampler: QuerySampler = StreamQuerySampler()

    experiment_modules: ExperimentModules = StreamExperiment(
        query_selector=StreamQuerySelector(
            query_optimizer=NoQueryOptimizer(query_sampler=StreamQuerySampler(), selection_criteria=AllSelectionCriteria()),
            query_decider=ThresholdQueryDecider(threshold=0.0),
            ),
        estimator=PassThroughEstimator(),
    )

    evaluators: Iterable[Evaluator] = (PrintExpTimeEvaluator(), LogOracleEvaluator(), LogAllEvaluator(), LogTVPGTEvaluator(), EstimatorEvaluator())

