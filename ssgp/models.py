import math
import torch
import ghalton as gh
from . import ff_frequencies
from . import mean_functions
from . import util


def cos_sin(X, S):
    """
       Computes cosine-sine random Fourier features
       :param X: (N,D) inputs
       :param S: (M,D) frequencies
       :return: M-by-N matrix of features for each input
    """
    M, D = S.shape
    dot_products = torch.mm(S,X.t())
    return torch.cat((torch.cos(dot_products), torch.sin(dot_products))) / math.sqrt(M)


class ISSGPR(object):
    """
    Sparse-spectrum Gaussian process regression implementing incremental updates according to:
    
    Gijsberts, Arjan, and Giorgio Metta. 2013. “Real-Time Model Learning Using Incremental Sparse Spectrum Gaussian Process Regression.” Neural Networks 41: 59–69.
    """
    kernel_samplers = {"squared_exponential": ff_frequencies.standard_normal,
                       "matern1": ff_frequencies.matern_12,
                       "matern3": ff_frequencies.matern_32,
                       "matern5": ff_frequencies.matern_52}
    def __init__(self,
                 n_frequencies,
                 dim,
                 kernel_type = list(kernel_samplers.keys())[0],
                 noise_stddev = 1e-4,
                 lengthscale = 1.,
                 signal_stddev = 1.,
                 mean_function = None
                ):
        super(ISSGPR,self).__init__()
        self.dtype = torch.float32

        self.dim = dim
        self.n_frequencies = n_frequencies
        self.noise_stddev = torch.as_tensor(noise_stddev)
        
        perm = gh.EA_PERMS[:dim]
        self.sequencer = gh.GeneralizedHalton(perm)
        base_freqs = util.ensure_torch(self.sequencer.get(int(n_frequencies)))
        
        self.raw_spec = ISSGPR.kernel_samplers[kernel_type](base_freqs)
        self._set_lengthscale(lengthscale)
        self.signal_stddev = util.ensure_torch(signal_stddev)
        self.clear_data()
        self.X = None
        self.Y = None
        if mean_function is None:
            mean_function = mean_functions.ZeroMean()
        self.mean_function = mean_function

        
    def get_dimensionality(self):
        return self.dim
    
    def _set_lengthscale(self,value):
        self._lengthscale = util.ensure_torch(value)
        self.spec = self.raw_spec/self._lengthscale
        
    def get_lengthscale(self):
        return self._lengthscale
        
    def _set_noise_stddev(self,value):
        self.noise_stddev = util.ensure_torch(value)
        
    def _set_signal_stddev(self,value):
        self.signal_stddev = util.ensure_torch(value)
        
    def set_hyperparameters(self,lengthscale,signal_stddev=None,noise_stddev=None,mean_params=None):
        self._set_lengthscale(lengthscale)
        if signal_stddev is not None:
            self._set_signal_stddev(signal_stddev)
        if noise_stddev is not None:
            self._set_noise_stddev(noise_stddev)
        if mean_params is not None:
            self.mean_function.set_parameters(mean_params)
            
        if self.Y is None:
            self.clear_data() # updates internals even if there is no data
        else:
            self.set_data(self.X,self.Y) # recomputes internals
            
    def get_hyperparameters(self):
        return torch.stack((self._lengthscale,self.signal_stddev,self.noise_stddev,self.mean_function.get_parameters()))
        
    def clear_data(self):
        self.training_mat = (self.noise_stddev/self.signal_stddev)*torch.eye(self.n_frequencies*2) # Cholesky
        self.training_vec = torch.zeros((self.n_frequencies*2,1))
        self._update_weights()

    def feature_transform(self,X):
        return self.signal_stddev*cos_sin(X,self.spec)
    
    def updated_training_mat(self,feature):
        return torch.qr(torch.cat((self.training_mat,feature.t()),dim=0))[1]

    def _update_dataset(self,x,y):
        if self.X is None:
            self.X = x
        else:
            self.X = torch.cat((self.X,x),dim=0)
        if self.Y is None:
            self.Y = torch.full((1,1),y.item())
        else:
            self.Y = torch.cat((self.Y,torch.full((1,1),y.item())),dim=0)

    def update(self,x,y):
        assert x.dim() == 2, "Data point should be a row vector"
        assert y.dim() == 0, "Observation value should be a scalar"
        phi_t = self.feature_transform(x)
        self.training_vec += phi_t*(y-self.mean_function(x).squeeze())
        self.training_mat = self.updated_training_mat(phi_t)
        self._update_weights()
        self._update_dataset(x,y)
        
    def _update_weights(self):
        self.weights_train = torch.cholesky_solve(self.training_vec,self.training_mat,upper=True) # not to be confused with BLR weights sample
    
    def set_data(self,X_all,Y_all):
        assert Y_all.dim() == 2 and X_all.dim() == 2, "Invalid data tensor dimensions. Both X and Y should be matrices."
        features = self.feature_transform(X_all)
        self.training_vec = torch.matmul(features,(Y_all-self.mean_function(X_all)))
        mat_A = torch.matmul(features,features.t())+(self.noise_stddev**2)*torch.eye(features.size(0))
        self.training_mat = torch.cholesky(mat_A,upper=True)
        self._update_weights()
        self.X = X_all
        self.Y = Y_all
        
    def evaluate_nlml(self,lengthscale,signal_stddev=None,noise_stddev=None,mean_params=None,X=None,Y=None,enforce_constraints=False,verbose=False):
        """
        Computes the log marginal likelihood of the model's hyper-parameters with respect to the data
        """
        if signal_stddev is None:
            signal_stddev = self.signal_stddev
        if noise_stddev is None:
            noise_stddev = self.noise_stddev
        if mean_params is None:
            mean_params = self.mean_function.get_parameters()
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
            
        if enforce_constraints:
            lengthscale = util.ensure_positive(lengthscale)
            signal_stddev = util.ensure_positive(signal_stddev)
            noise_stddev = util.ensure_positive(noise_stddev)
            
        if verbose:
            print("evaluate_nlml:",lengthscale,signal_stddev,noise_stddev,mean_params)

        n_freqs = self.raw_spec.shape[0]
        n_data = Y.shape[0]
        features = signal_stddev*cos_sin(X,self.raw_spec/lengthscale)
        Y_diff = Y-self.mean_function(X,params=mean_params)
        training_vec = torch.matmul(features,Y_diff)
        mat_A = torch.matmul(features,features.t())+(noise_stddev**2)*torch.eye(features.shape[0])
        weights_train,_ = torch.solve(training_vec,mat_A)
        lml_1_1 = torch.mm(Y_diff.t(),Y_diff)
        lml_1_2 = torch.mm(weights_train.t(),training_vec)
        lml_1 = -(lml_1_1-lml_1_2)/(2*noise_stddev**2)
        lml_2 = -torch.log(torch.cholesky(mat_A).diagonal()).sum()
        lml = lml_1 + lml_2 + n_freqs*torch.log(noise_stddev) - 0.5*n_data*torch.log(2*math.pi*noise_stddev**2)
        return -lml
        
    def predict(self,X_test):
        phi_test = self.feature_transform(X_test)
        prior_mean = self.mean_function(X_test)
        mean_test = prior_mean + torch.matmul(phi_test.t(),self.weights_train)
        covar_test = (self.noise_stddev**2)*torch.matmul(phi_test.t(),torch.cholesky_solve(phi_test,self.training_mat,upper=True))

        return mean_test,covar_test
