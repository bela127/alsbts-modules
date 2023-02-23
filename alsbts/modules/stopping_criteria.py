from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alts.core.stopping_criteria import StoppingCriteria

if TYPE_CHECKING:
    from typing import Tuple, List

@dataclass
class DataExhaustedStoppingCriteria(StoppingCriteria):

    def next(self, iteration):
        if self.exp.oracle.data_source.exhausted:
            return False
        return True