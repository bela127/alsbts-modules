from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.stopping_criteria import StoppingCriteria

from alvsts.modules.experiment_setup import Simulation

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class SimEndStoppingCriteria(StoppingCriteria):

    def next(self, iteration):
        if self.exp.oracle.data_source.exhausted:
            self.exp.oracle.data_source.exp_setup.stop_simulation()
            return False
        return True