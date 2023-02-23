from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

from alsbts.core.estimator import Estimator
from alts.core.configuration import pre_init, post_init, init
import numpy as np

#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import GPy

from GPy.kern import Kern, RBF, Bias, White
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from nptyping import  NDArray, Number, Shape

@dataclass
class PassThroughEstimator(Estimator):

    def estimate(self, times, queries, vars) -> NDArray[Shape["query_nr, ... result_dim"], Number]:
        if self.exp_modules.result_data_pool.last_results.shape[0] > 0:
            estimation = self.exp_modules.result_data_pool.last_results[-1][..., None]
        else:
            estimation = np.zeros((queries.shape[0],1))
        return estimation


@dataclass
class GPEstimator(Estimator):
    length_scale: float = 0.5
    noise:float = 0.00#1

    gaussian_process: GPy.models.GPRegression = pre_init(default=None)

    newly_trained: bool = pre_init(default=False)

    kern: GPy.kern.Kern = init(default=None)


    def __post_init__(self):
        super().__post_init__()
        if self.kern is None:
            self.kern = RBF(lengthscale=self.length_scale, input_dim=1, variance=0.25, active_dims=[1])

    def train(self):
        all_queries = self.exp_modules.result_data_pool.queries
        all_results = self.exp_modules.result_data_pool.results
        self.gaussian_process = GPy.models.GPRegression(all_queries, all_results, self.kern, noise_var=self.noise)
        self.apply_constrains()
        self.newly_trained = True
        

    def estimate(self, times, queries, vars) -> NDArray[Shape["query_nr, ... result_dim"], Number]:
                
        if self.newly_trained:
            self.newly_trained = False
            last_results = self.exp_modules.result_data_pool.last_results
            estimation = last_results
            return estimation
            
        times = self.exp_modules.stream_data_pool.last_queries
        vars = self.exp_modules.stream_data_pool.last_results
        queries = np.concatenate((times, vars), axis=1)
        queries, estimation, estimation_cov = self.query(queries)
        
        return estimation
    
    def apply_constrains(self):
        self.gaussian_process.rbf.lengthscale.fix()
        self.gaussian_process.Gaussian_noise.variance.fix()
        self.gaussian_process.rbf.variance.fix()
        #self.gaussian_process.optimize_restarts(num_restarts=3, max_iters=1000, messages=False, ipython_notebook=False)
    
    def query(self, queries):
        if self.gaussian_process is not None:
            estimation, estimation_cov = self.gaussian_process.predict_noiseless(queries)
        else:
            estimation = np.zeros((queries.shape[0],1))
            estimation_cov = np.ones((queries.shape[0],1)) * 100
        return queries, estimation, estimation_cov

@dataclass
class BrownGPEstimator(GPEstimator):

    def __post_init__(self):
        self.kern = Combined(rbf_lengthscale=self.length_scale, rbf_variance=0.25, brown_variance=0.005)
        super().__post_init__()

    def apply_constrains(self):
        self.gaussian_process.Gaussian_noise.variance.fix()
        self.gaussian_process.Combined.rbf_variance.fix()
        self.gaussian_process.Combined.rbf_lengthscale.fix()
        self.gaussian_process.Combined.brown_variance.fix()
        #self.gaussian_process.optimize_restarts(num_restarts=3, max_iters=1000, messages=False, ipython_notebook=False)


@dataclass
class BrownGPAdaptEstimator(GPEstimator):

    def __post_init__(self):
        self.kern = Combined(rbf_lengthscale=self.length_scale, rbf_variance=0.25, brown_variance=0.005)
        super().__post_init__()

    def apply_constrains(self):
        self.gaussian_process.Gaussian_noise.variance.fix()
        self.gaussian_process.Combined.rbf_variance.fix()
        self.gaussian_process.Combined.rbf_lengthscale.fix()
        #self.gaussian_process.Combined.brown_variance.fix()
        self.gaussian_process.optimize_restarts(num_restarts=3, max_iters=1000, messages=False, ipython_notebook=False)


@dataclass
class IntBrownGPEstimator(GPEstimator):

    def __post_init__(self):
        self.kern = IntCombined(rbf_lengthscale=self.length_scale, rbf_variance=0.25, brown_variance=0.005)
        super().__post_init__()

    def train(self):
        all_query_results = self.exp_modules.process_data_pool.queries
        all_queries = self.exp_modules.result_data_pool.queries
        all_results = self.exp_modules.result_data_pool.results
        queries = np.concatenate((all_query_results[:,:1], all_queries), axis=1)
        self.gaussian_process = GPy.models.GPRegression(queries, all_results, self.kern, noise_var=self.noise)
        self.apply_constrains()
        self.newly_trained = True

    def query(self, queries):
        if self.gaussian_process is not None:
            actual_queries = np.concatenate((queries[:,:1],queries), axis=1)
            estimation, estimation_cov = self.gaussian_process.predict_noiseless(actual_queries)
        else:
            estimation = np.zeros((queries.shape[0],1))
            estimation_cov = np.ones((queries.shape[0],1)) * 100
        return queries, estimation, estimation_cov

    def apply_constrains(self):
        self.gaussian_process.Gaussian_noise.variance.fix()
        self.gaussian_process.IntCombined.rbf_variance.fix()
        self.gaussian_process.IntCombined.rbf_lengthscale.fix()
        self.gaussian_process.IntCombined.brown_variance.fix()
        #self.gaussian_process.optimize_restarts(num_restarts=3, max_iters=1000, messages=False, ipython_notebook=False)


class Combined(Kern):
    """
    Abstract class for change kernels
    """
    def __init__(self, input_dim = 2, active_dims=None,  rbf_variance = 10, rbf_lengthscale = 0.4, brown_variance = 10, name = 'Combined'):

        super().__init__(input_dim, active_dims, name)

        self.brown = GPy.kern.Brownian(variance=brown_variance, active_dims=[0])
        self.rbf = GPy.kern.RBF(variance=rbf_variance,lengthscale=rbf_lengthscale, input_dim=1, active_dims=[1])
        self.rbf_add = GPy.kern.RBF(variance=rbf_variance,lengthscale=rbf_lengthscale, input_dim=1, active_dims=[1])

        self.rbf_variance = Param('rbf_variance', rbf_variance, Logexp())
        self.link_parameter(self.rbf_variance)
        self.rbf_lengthscale = Param('rbf_lengthscale', rbf_lengthscale, Logexp())
        self.link_parameter(self.rbf_lengthscale)
        self.brown_variance = Param('brown_variance', brown_variance, Logexp())
        self.link_parameter(self.brown_variance)

    def parameters_changed(self):
        self.rbf.variance = self.rbf_add.variance = self.rbf_variance
        self.rbf.lengthscale = self.rbf_add.lengthscale = self.rbf_lengthscale
        self.brown.variance = self.brown_variance

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        return self.rbf_add.K(X, X2) + self.brown.K(X, X2) * self.rbf.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        return self.rbf_add.Kdiag(X) + self.brown.Kdiag(X) * self.rbf.Kdiag(X)

    # NOTE ON OPTIMISATION:
    #   Should be able to get away with only optimising the parameters of one sigmoidal kernel and propagating them

    def update_gradients_full(self, dL_dK, X, X2 = None): # See NOTE ON OPTIMISATION
        self.brown.update_gradients_full(dL_dK * self.rbf.K(X, X2), X, X2)
        self.rbf.update_gradients_full(dL_dK * self.brown.K(X, X2), X, X2)

        self.rbf_add.update_gradients_full(dL_dK, X, X2)

        self.rbf_variance.gradient = self.rbf.variance.gradient + self.rbf_add.variance.gradient
        self.rbf_lengthscale.gradient = self.rbf.lengthscale.gradient + self.rbf_add.lengthscale.gradient
        self.brown_variance.gradient = self.brown.variance.gradient


    def update_gradients_diag(self, dL_dK, X):
        self.brown.update_gradients_diag(dL_dK * self.rbf.Kdiag(X), X)
        self.rbf.update_gradients_diag(dL_dK * self.brown.Kdiag(X), X)

        self.rbf_add.update_gradients_diag(dL_dK, X)

        self.rbf_variance.gradient = self.rbf.variance.gradient + self.rbf_add.variance.gradient
        self.rbf_lengthscale.gradient = self.rbf.lengthscale.gradient + self.rbf_add.lengthscale.gradient
        self.brown_variance.gradient = self.brown.variance.gradient



def plot_3d(self, title='Estimated Model'):
    train = np.concatenate((self.times, self.rvss), axis=1)

    min_x = -1
    max_x = 4
    stop_time = 900

    nr_plot_points = 100

    gp = self.gaussian_process
    X = np.linspace(min_x, max_x, num=nr_plot_points)
    t = np.linspace(0, stop_time*1.2, num=nr_plot_points)
    X_pred, t_pred = np.meshgrid(X, t)
    pred = np.concatenate((t_pred.reshape((-1,1)), X_pred.reshape((-1,1))), axis=1)

    Ypred,YpredCov = gp.predict_noiseless(pred)
    SE = np.sqrt(YpredCov)[:,0]
    Y_pred = Ypred.reshape(X_pred.shape)
    SE_pred = SE.reshape(X_pred.shape)

    color_dimension = SE_pred*1.96
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    fcolors[:,:,3] = 0.2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')




    ax.plot_surface(t_pred, X_pred, Y_pred, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

    ax.scatter(train[:,0], train[:,1], self.vss,label='GT Data', )
    for t in train:
        ax.plot([t[0],t[0]],[t[1],t[1]],[0,1])
    ax.scatter(stop_time * 1.2 * np.ones_like(train[:,0]), train[:,1], self.vss,label='GT Data', )

    ax.set_xlabel('time')
    ax.set_ylabel('rvs')
    ax.set_zlabel('vs')

    plt.title(title)
    plt.legend()
    plt.colorbar(m)
    plt.show()



class IntegralBrown(Kern): 
    """
    Integral kernel. This kernel allows 1d histogram or binned data to be modelled.
    The outputs are the counts in each bin. The inputs (on two dimensions) are the start and end points of each bin.
    The kernel's predictions are the latent function which might have generated those binned results.
    """

    def __init__(self, input_dim = 2, variance=1, ARD=False, active_dims=None, name='integral'):
        """
        """
        super(IntegralBrown, self).__init__(input_dim, active_dims, name)
 
        self.variance = Param('variance', variance, Logexp()) #Logexp - transforms to allow positive only values...
        self.link_parameters(self.variance) #this just takes a list of parameters we need to optimise.


    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:  #we're finding dK_xx/dTheta
            dK_dv = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    dK_dv[i,j] = self.k_FF(x[1],x2[1],x[0],x2[0])  #the gradient wrt the variance is k_FF.
            self.variance.gradient = np.sum(dK_dv * dL_dK)
        else:     #we're finding dK_xf/Dtheta
            raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")


    def xx(self, yl,yu,xl,xu):
        return 1/2*xu**2*yu-1/2*xl**2*yu-1/2*xu**2*yl+1/2*xl**2*yl

    def yy(self, yl,yu,xl,xu):
        return self.xx(xl,xu, yl,yu)
    
    def xy(self, l,u):
        #1/4 pyramid + the below cuboid
        return 1/3*(u-l)**3+(u-l)**2*l

    def x(self, x, l, u):
        return x*u-x*l
    
    def y(self, l, u):
        return 1/2*u**2-1/2*l**2

    def k_FF(self,t,tprime,s,sprime):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""

        if s <= sprime < tprime <= t:
            xx = self.xx(tprime, t, sprime, tprime)
            xy = self.xy(sprime,tprime)
            yy= self.yy(s,sprime,sprime,tprime)
            i = xx + xy + yy
        elif sprime < tprime <= s < t:
            i = self.xx(s,t,sprime,tprime)
        elif sprime <= s < tprime <= t:
            xx = self.xx(s, t, sprime, s) + self.xx(tprime, t, s, tprime)
            xy = self.xy(s, tprime)
            i = xx + xy
        elif s <= sprime < t <= tprime:
            yy = self.yy(s, sprime,sprime,tprime) + self.yy(sprime, t, t, tprime)
            xy = self.xy(sprime, t)
            i = yy + xy
        elif s < t <= sprime < tprime:
            i = self.yy(s, t, sprime, tprime)
        elif sprime <= s < t <= tprime:
            xx = self.xx(s, t, sprime, s)
            xy = self.xy(s,t)
            yy = self.yy(s,t, t, tprime)
            i= xx + xy + yy
        else:
            #TODO: RuntimeError: This should never happen, i guess i should check the code, please report: (0.0, 0.0, 0.0, 0.0)
            raise RuntimeError(f"This should never happen, i guess i should check the code, please report: {s,t,sprime, tprime}")
        return i


    def k_ff(self,x,y):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        return np.fmin(x,y)


    def k_Ff(self,t,x,s):
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
        #      First integral *-* + Second Integral *-*

        if x <= s < t:
            i = self.x(x, s, t)
        elif s < x < t:
            y = self.y(s,x)
            x = self.x(x,x,t)
            i= x + y
        elif s < t <= x:
            i = self.y(s,t)
        else:
            raise RuntimeError(f"This should never happen, i guess i should check the code, please report: {s,x,t}")
        return i


        #return 1/2*np.fmin(x,t)**2-1/2*np.fmin(s,x)**2 + np.fmin(x,t)*t-np.fmin(x,t)**2


    def K(self, X, X2=None):
        """Note: We have a latent function and an output function. We want to be able to find:
          - the covariance between values of the output function
          - the covariance between values of the latent function
          - the "cross covariance" between values of the output function and the latent function
        This method is used by GPy to either get the covariance between the outputs (K_xx) or
        is used to get the cross covariance (between the latent function and the outputs (K_xf).
        We take advantage of the places where this function is used:
         - if X2 is none, then we know that the items being compared (to get the covariance for)
         are going to be both from the OUTPUT FUNCTION.
         - if X2 is not none, then we know that the items being compared are from two different
         sets (the OUTPUT FUNCTION and the LATENT FUNCTION).
        
        If we want the covariance between values of the LATENT FUNCTION, we take advantage of
        the fact that we only need that when we do prediction, and this only calls Kdiag (not K).
        So the covariance between LATENT FUNCTIONS is available from Kdiag.        
        """
        if X2 is None:
            K_FF = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    K_FF[i,j] = self.k_FF(x[1],x2[1],x[0],x2[0])
            eigv = np.linalg.eig(K_FF)
            return K_FF * self.variance
        else:
            K_Ff = np.zeros([X.shape[0],X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    K_Ff[i,j] = self.k_Ff(x[1],x2[1],x[0]) #x2[0] unused, see k_Ff docstring for explanation.
            return K_Ff * self.variance


    def Kdiag(self, X):
        """I've used the fact that we call this method during prediction (instead of K). When we
        do prediction we want to know the covariance between LATENT FUNCTIONS (K_ff) (as that's probably
        what the user wants).
        $K_{ff}^{post} = K_{ff} - K_{fx} K_{xx}^{-1} K_{xf}$"""
        K_ff = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            K_ff[i] = self.k_ff(x[1],x[1])
        return K_ff * self.variance

class IntCombined(Kern):
    """
    Abstract class for change kernels
    """
    def __init__(self, input_dim = 3, active_dims=None,  rbf_variance = 10, rbf_lengthscale = 0.4, brown_variance = 10, name = 'IntCombined'):

        super().__init__(input_dim, active_dims, name)

        self.brown = IntegralBrown(variance=brown_variance, active_dims=[0,1])
        self.rbf = GPy.kern.RBF(variance=rbf_variance,lengthscale=rbf_lengthscale, input_dim=1, active_dims=[2])
        self.rbf_add = GPy.kern.RBF(variance=rbf_variance,lengthscale=rbf_lengthscale, input_dim=1, active_dims=[2])

        self.rbf_variance = Param('rbf_variance', rbf_variance, Logexp())
        self.link_parameter(self.rbf_variance)
        self.rbf_lengthscale = Param('rbf_lengthscale', rbf_lengthscale, Logexp())
        self.link_parameter(self.rbf_lengthscale)
        self.brown_variance = Param('brown_variance', brown_variance, Logexp())
        self.link_parameter(self.brown_variance)

    def parameters_changed(self):
        self.rbf.variance = self.rbf_add.variance = self.rbf_variance
        self.rbf.lengthscale = self.rbf_add.lengthscale = self.rbf_lengthscale
        self.brown.variance = self.brown_variance

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        return self.rbf_add.K(X, X2) + self.brown.K(X, X2) * self.rbf.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        return self.rbf_add.Kdiag(X) + self.brown.Kdiag(X) * self.rbf.Kdiag(X)

    # NOTE ON OPTIMISATION:
    #   Should be able to get away with only optimising the parameters of one sigmoidal kernel and propagating them

    def update_gradients_full(self, dL_dK, X, X2 = None): # See NOTE ON OPTIMISATION
        self.brown.update_gradients_full(dL_dK * self.rbf.K(X, X2), X, X2)
        self.rbf.update_gradients_full(dL_dK * self.brown.K(X, X2), X, X2)

        self.rbf_add.update_gradients_full(dL_dK, X, X2)

        self.rbf_variance.gradient = self.rbf.variance.gradient + self.rbf_add.variance.gradient
        self.rbf_lengthscale.gradient = self.rbf.lengthscale.gradient + self.rbf_add.lengthscale.gradient
        self.brown_variance.gradient = self.brown.variance.gradient


    def update_gradients_diag(self, dL_dK, X):
        self.brown.update_gradients_diag(dL_dK * self.rbf.Kdiag(X), X)
        self.rbf.update_gradients_diag(dL_dK * self.brown.Kdiag(X), X)

        self.rbf_add.update_gradients_diag(dL_dK, X)

        self.rbf_variance.gradient = self.rbf.variance.gradient + self.rbf_add.variance.gradient
        self.rbf_lengthscale.gradient = self.rbf.lengthscale.gradient + self.rbf_add.lengthscale.gradient
        self.brown_variance.gradient = self.brown.variance.gradient
