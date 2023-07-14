from __future__ import annotations
from typing import TYPE_CHECKING

from alts.core.query.query_selector import QuerySelector

if TYPE_CHECKING:
    from alts.core.subscribable import Subscribable

class StreamQuerySelector(QuerySelector):
    
    def stream_update(self, subscription: Subscribable):
        self.decide()
        return super().stream_update(subscription)

    def update(self, subscription: Subscribable) -> None:
        return super().update(subscription)

