from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alts.core.query.query_decider import QueryDecider
from alts.core.configuration import Required, is_set, init

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Tuple, Optional
    from nptyping import NDArray, Number, Shape
    from alts.core.experiment_modules import ExperimentModules

@dataclass
class EmptyQueryDecider(QueryDecider):
    query_decider: QueryDecider = init()

    def post_init(self):
        super().post_init()
        self.query_decider = self.query_decider(exp_modules = self.exp_modules)

    def decide(self, query_candidates: NDArray[Shape["query_nr, ... query_dims"], Number], scores: NDArray[Shape["query_nr, [query_score]"], Number]) -> Tuple[bool, NDArray[Shape["query_nr, []"], Number]]:
        flag, queries = self.query_decider.decide(query_candidates, scores)
        query = np.empty(shape=(queries.shape[0],0))
        return flag, query
    