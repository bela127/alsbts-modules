from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from scipy.stats import kruskal #type: ignore

from aldtts.core.multi_sample_test import MultiSampleTest

if TYPE_CHECKING:
    ...


@dataclass
class KWHMultiSampleTest(MultiSampleTest):
    
    def test(self, samples):
        t, p = kruskal(*samples)
        return t, p