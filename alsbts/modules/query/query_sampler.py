from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alts.core.query.query_sampler import QuerySampler


if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from typing import Optional
    from nptyping import NDArray, Number, Shape

@dataclass
class StreamQuerySampler(QuerySampler):

    def sample(self, num_queries: Optional[int] = None) -> NDArray[Shape["query_nr, ... query_dims"], Number]:
        query = np.concatenate((self.exp_modules.stream_data_pool.last_queries, self.exp_modules.stream_data_pool.last_results), axis=1)
        return query

