from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np

class UKFModel:
    initial_vs = np.array([1.0])
    initial_uncertainty = 10
    model: TransitionModel = None


    def __init__(self) -> None:
        # create sigma points to use in the filter. This is standard for Gaussian processes
        self.sigma_points = MerweScaledSigmaPoints(1, alpha=0.1, beta=2.0, kappa=0)

        self.kf = UKF(dim_x=1, dim_z=1, dt=None, fx=self.fx, hx=self.hx, points=self.points)

        # x: initial state
        #   x0: voltage sensitivity
        self.kf.x = self.initial_vs
        # initial uncertainty (covariance estimate matrix)
        self.kf.P *= self.initial_uncertainty
        # initial measurement noise matrix (eye(dim_z) per default)
        # measurement uncertainty: low values -> high trust in measurements
        self.kf.R *= 10 ** -5
        # process noise matrix (eye(dim_x) per default)
        # kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.03)
    
    def fx(self, x, dt, currentRelativeVS):
        # state transition function - predict next state based on model

        input_batch = np.asarray([[x]])
        xNew = self.model.predict(input_batch)[0]
        return xNew

    def hx(self, x):
        # measurement function - convert state into a measurement
        return x
        
        