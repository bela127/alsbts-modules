from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np

from alts.core.query.query_sampler import QuerySampler
from alts.core.data.data_pools import StreamDataPools



if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from typing import Optional
    from nptyping import NDArray, Number, Shape

@dataclass
class StreamQuerySampler(QuerySampler):

    def post_init(self):
        super().post_init()
        if not isinstance(self.data_pools, StreamDataPools):
            raise TypeError("StreamQuerySampler requires StreamDataPools")

    @property
    def data_pools(self) -> StreamDataPools:
        return super().data_pools

    def sample(self, num_queries: Optional[int] = None) -> NDArray[Shape["query_nr, ... query_dims"], Number]:
        query = np.concatenate((self.data_pools.stream.last_queries, self.data_pools.stream.last_results), axis=1)
        return query

