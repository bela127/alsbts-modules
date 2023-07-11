from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

from alts.core.configuration import pre_init
from alsbts.core.change_detector import ChangeDetector

@dataclass
class OptimalChangeDetector(ChangeDetector):
    last_value: float = pre_init(default= 0.0)

    def detect(self, changing_signal):
        change_scores = np.abs(changing_signal - self.last_value)
        self.last_value = changing_signal[-1, 0]
        return change_scores

@dataclass
class NoisyChangeDetector(ChangeDetector):
    change_offset_std: float = 5
    wrong_detection_ratio: float = 0.005
    missed_detection_ratio: float = 0.025
    time_step: float = 1 #0.05

    gt_change = False
    last_value = 0
    offset_steps = 0
    

    def __post_init__(self):
        self._change_kinds = []

    def detect(self, changing_signal):
        changes = np.empty_like(changing_signal)

        for i, changing_value in enumerate(changing_signal):
            if not self.gt_change and self.last_value != changing_value:
                self.last_value = changing_value
                self.gt_change = True
                self.offset_steps = np.abs(np.random.normal(scale=self.change_offset_std))
            
            if self.gt_change:
                self.offset_steps -= self.exp_modules.time_source.time_step
                
            if self.gt_change and self.offset_steps <= 0:
                self.gt_change = False

                if np.random.uniform() > self.missed_detection_ratio:
                    self.change_kind(1) #True detect
                    changes[i, 0] = 1 #detect change
                    continue
                else:
                    self.change_kind(-2) #missed detect
                    changes[i, 0] =  0 #do not detect change, even if there should be a detection
                    continue
            
            if np.random.uniform() < self.wrong_detection_ratio:
                self.change_kind(-1) #False detect
                changes[i, 0] =  1 #detect change, even if there is no change
                continue

            self.change_kind(0) #no change
            changes[i, 0] =  0 #no change
            continue
        return changes

    @property
    def change_kinds(self):
        return np.asarray(self._change_kinds)

    def change_kind(self, kind):
        self._change_kinds.append(kind)
