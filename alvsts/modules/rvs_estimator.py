

from dataclasses import dataclass
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


from alts.core.configuration import Configurable


class RVSEstimator(Configurable):

    def estimate(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput) -> float:
        raise NotImplementedError()

@dataclass
class OptimalRVSEstimator(RVSEstimator):

    def estimate(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput):
        return vs_gt

@dataclass
class NoisyGaussianRVSEstimator(RVSEstimator):
    noise_var: float = 0.3
    length_scale: float = 1
    probability_of_function_change:float = 0.05
    change_size_proportion: float = 0.25


    vs_offsets = np.empty((0,1))
    gt_vss = np.empty((0,1))

    def __post_init__(self):
        self._change_time = []
        self.gaussian_process = GaussianProcessRegressor(kernel=RBF(length_scale= self.length_scale, length_scale_bounds='fixed'), random_state=None)

    def estimate(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput):

        #reduce training set size for faster training
        xy = np.concatenate((self.gt_vss, self.vs_offsets), axis=1)
        xy = np.unique(xy, axis=0)
        self.gt_vss = xy[:, 0][:,None]
        self.vs_offsets = xy[:, 1][:,None]

        if self.gt_vss.size > 0:
            #remove part of the training set if the function should change
            if self.gt_vss[0] != vs_gt and np.random.uniform() < self.probability_of_function_change:
                self.change_time(timeOutput)
                self.gaussian_process = GaussianProcessRegressor(kernel=RBF(length_scale= self.length_scale, length_scale_bounds='fixed'), random_state=None)
                
                unchanged_size = int(np.floor((1 - self.change_size_proportion) * len(self.gt_vss)))
                index = np.argsort(self.gt_vss[:,0])
                start_index = np.random.randint(len(self.gt_vss))

                #self.gt_vss = np.tile(self.gt_vss[index], (2,1))[start_index:start_index+unchanged_size]
                #self.vs_offsets = np.tile(self.vs_offsets[index], (2,1))[start_index:start_index+unchanged_size]

                self.gt_vss =self.gt_vss[index][0][None,:]
                self.vs_offsets = self.vs_offsets[index][0][None,:]
            #adapt to new training set
            self.gaussian_process.fit(self.gt_vss, self.vs_offsets[:,0])

        #sample from model and add sample to training set
        vs_offset = self.gaussian_process.sample_y([[vs_gt]], random_state=None)
        self.vs_offsets = np.concatenate((self.vs_offsets, vs_offset))
        self.gt_vss = np.concatenate((self.gt_vss, np.asarray([[vs_gt]])))

        noise = np.random.normal(scale=self.noise_var)
        return vs_gt + vs_offset[0] + noise

    @property
    def change_times(self):
        return np.asarray(self._change_time)

    def change_time(self, time):
        self._change_time.append(time)