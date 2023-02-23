
from alts.core.query.query_selector import QuerySelector

class StreamQuerySelector(QuerySelector):
    
    def stream_update(self):
        self.decide()
        return super().stream_update()

    def update(self) -> None:
        return super().update()

