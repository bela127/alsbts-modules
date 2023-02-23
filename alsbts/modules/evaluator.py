from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from typing_extensions import Self
from alts.core.evaluator import LogingEvaluator, Evaluate

import numpy as np
from matplotlib import pyplot as plot # type: ignore
import os

from alsbts.core.experiment_modules import StreamExperiment


if TYPE_CHECKING:
    from typing import List, Tuple
    from alts.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape

@dataclass
class EstimatorEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "estimation_data"

    def register(self, experiment: Experiment):
        super().register(experiment)

        if not isinstance(self.experiment.experiment_modules, StreamExperiment):
            raise ValueError("for this evaluator the Process needs to be a DataSourceProcess")

        self.experiment.experiment_modules.estimator.estimate = Evaluate(self.experiment.experiment_modules.estimator.estimate)
        self.experiment.experiment_modules.estimator.estimate.post(self.save_estimation)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.estimation: NDArray[Shape["query_nr, ... result_dim"], Number] = None

    def save_estimation(self, estimation: NDArray[Shape["query_nr, ... result_dim"], Number]):

        if self.estimation is None:
            self.estimation = estimation
        else:
           self.estimation = np.concatenate((self.estimation, estimation))
    
    def log_data(self, exp_nr):
        np.save(f'{self.path}/{self.file_name}.npy', self.estimation)