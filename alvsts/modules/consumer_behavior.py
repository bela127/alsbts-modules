from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plot # type: ignore
import os


if TYPE_CHECKING:
    from typing import Tuple
    from nptyping import  NDArray, Number, Shape

class ConsumerBehavior():

    def behavior(self) -> Tuple[NDArray[Number, Shape["change_times"]], NDArray[Number, Shape["kps"]]]:
        raise NotImplementedError()

@dataclass
class EquidistantTimeUniformKpBehavior(ConsumerBehavior):
    numberOfLoadChanges = 60
    lower_kp=0
    uper_kp=2
    start_time = 9
    end_time = 599

    def behavior(self) -> Tuple[int, int]:
        kp = np.random.uniform(self.lower_kp, self.uper_kp, (self.numberOfLoadChanges + 1))
        change_time = np.insert(np.linspace(self.start_time, self.end_time, self.numberOfLoadChanges), 0, [0])
        return change_time, kp

@dataclass
class RandomTimeUniformKpBehavior(ConsumerBehavior):
    numberOfLoadChanges = 60
    lower_kp=0
    uper_kp=2
    start_time = 1
    end_time = 600

    def behavior(self) -> Tuple[int, int]:
        kp = np.random.uniform(self.lower_kp, self.uper_kp, (self.numberOfLoadChanges + 1))
        change_time = np.insert(np.sort(np.random.uniform(self.start_time, self.end_time, self.numberOfLoadChanges)), 0, [0])
        return change_time, kp

