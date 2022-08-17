from dataclasses import dataclass
from typing import Literal, Union

import numpy as np


from alts.core.configuration import Configurable


class ChangeDetector(Configurable):

    def detect(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs) -> Literal[1, 0]:
        raise NotImplementedError()

@dataclass
class OptimalChangeDetector(ChangeDetector):
    last_rvs = 0

    def detect(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs):
        if self.last_rvs != vs_gt:
            self.last_rvs = vs_gt
            changed = 1 #detect change
        else:
            changed = 0 #no change
        return changed

@dataclass
class NoisyChangeDetector(ChangeDetector):
    change_offset_std: float = 5
    wrong_detection_ratio: float = 0.005
    missed_detection_ratio: float = 0.025

    gt_change = 0
    last_rvs = 0
    offset_steps = False
    

    def __post_init__(self):
        self._change_kind = []

    def detect(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs):
        if not self.gt_change and self.last_rvs != vs_gt:
            self.last_rvs = vs_gt
            self.gt_change = True
            self.offset_steps = np.abs(np.random.normal(scale=self.change_offset_std))
        
        if self.gt_change:
            self.offset_steps -= 1

        if self.gt_change and self.offset_steps <= 0:
            self.gt_change = False

            if np.random.uniform() > self.missed_detection_ratio:
                self.change_kind(1) #True detect
                return 1 #detect change
            else:
                self.change_kind(-2) #missed detect
                return 0 #do not detect change, even if there should be a detection
        
        if np.random.uniform() < self.wrong_detection_ratio:
            self.change_kind(-1) #False detect
            return 1 #detect change, even if there is no change

        self.change_kind(0) #no change
        return 0 #no change

    @property
    def change_kinds(self):
        return np.asarray(self._change_kind)

    def change_kind(self, kind):
        self._change_kind.append(kind)
