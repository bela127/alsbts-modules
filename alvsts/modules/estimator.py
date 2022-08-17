from alsbts.core.estimator import Estimator
import numpy as np

class PassThroughEstimator(Estimator):

    def estimate(self):
        
        vs = self.exp_modules.queried_data_pool.last_results[-1][8]
        return vs
