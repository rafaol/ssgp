import math
import torch
import ghalton as gh
from . import ff_frequencies
from . import mean_functions
from . import util


def cos_sin(X, S):
    """
       Computes cosine-sine Fourier features
       
       Parameters:
           X (torch.Tensor): N-by-D inputs
           S (torch.Tensor): m-by-D frequencies
       Returns:
           torch.Tensor: M-by-N matrix of features for the inputs, M=2m.
    """
    m, _ = S.shape
    dot_products = torch.mm(S, X.t())
    return torch.cat((torch.cos(dot_products), torch.sin(dot_products))) / math.sqrt(m)


class ISSGPR(object):
    """
    Sparse-spectrum Gaussian process regression implementing incremental updates according to:
    
    Gijsberts, Arjan, and Giorgio Metta. 2013. “Real-Time Model Learning Using Incremental Sparse Spectrum Gaussian
    Process Regression.” Neural Networks 41: 59–69.
    """
    kernel_samplers = {"squared_exponential": ff_frequencies.standard_normal,
                       "matern1": ff_frequencies.matern_12,
                       "matern3": ff_frequencies.matern_32,
                       "matern5": ff_frequencies.matern_52}

    def __init__(self,
                 n_frequencies,
                 dim,
                 kernel_type=list(kernel_samplers.keys())[0],
                 noise_stddev=1e-2,
                 lengthscale=1.,
                 signal_stddev=1.,
                 mean_function=None,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Constructor.
        
        Parameters:
            n_frequencies (int): Number of Fourier frequencies to generate

            dim (int): Dimensionality of the input domain

            kernel_type (str): String specifying which kernel to use (default: 'squared_exponential')

            noise_stddev (float or torch.Tensor): Standard deviation for the Gaussian observation noise model
            (default: 1e-4)

            lengthscale (float or torch.Tensor): Length-scale of the GP kernel (default: 1.0)

            signal_stddev (float or torch.Tensor): Signal standard deviation, i.e. a multiplicative scaling factor for
            the feature maps (default: 1.0)

            mean_function (ssgp.mean_functions.AbstractMeanFunction): GP prior mean function. If None, a zero-mean prior
            is used. (default: None)
        """
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.dim = dim
        self.n_frequencies = n_frequencies
        self.noise_stddev = self.ensure_torch(noise_stddev)

        perm = gh.EA_PERMS[:dim]
        self.sequencer = gh.GeneralizedHalton(perm)
        base_freqs = self.ensure_torch(self.sequencer.get(int(n_frequencies)))

        self.raw_spec = ISSGPR.kernel_samplers[kernel_type](base_freqs)
        self._set_lengthscale(lengthscale)
        self.signal_stddev = self.ensure_torch(signal_stddev)
        self.training_mat = None
        self.training_vec = None
        self.clear_data()
        if mean_function is None:
            mean_function = mean_functions.ZeroMean(dtype=self.dtype, device=self.device)
        self.mean_function = mean_function

    def ensure_torch(self, x):
        return util.ensure_torch(x, dtype=self.dtype, device=self.device)

    def get_dimensionality(self):
        """
        Retrieves inputs domain dimesionality set with the constructor.
        """
        return self.dim

    def _set_lengthscale(self, value):
        """
        Method used to internally set the kernel length-scale value. Not to be used directly externally.
        """
        self._lengthscale = self.ensure_torch(value)
        self.spec = self.raw_spec / self._lengthscale

    def get_lengthscale(self):
        """
        Retrieves the kernel length-scale.
        """
        return self._lengthscale

    def _set_noise_stddev(self, value):
        """
        Method used to internally set the observation noise model standard deviation. Not to be used directly
        externally.
        """
        self.noise_stddev = self.ensure_torch(value)

    def get_noise_stddev(self):
        return self.noise_stddev

    def _set_signal_stddev(self, value):
        """
        Method used to internally set the kernel signal standard deviation value. Not to be used directly externally.
        """
        self.signal_stddev = self.ensure_torch(value)

    def get_signal_stddev(self):
        return self.signal_stddev

    def set_hyperparameters(self, lengthscale, signal_stddev=None, noise_stddev=None, mean_params=None):
        """
        Method to set or update GP hyper-parameters.
        
        Parameters:
            lengthscale (float or torch.Tensor): Length-scale of the GP kernel.

            signal_stddev (float or torch.Tensor): Signal standard deviation, i.e. a multiplicative scaling factor for
            the feature maps. If None, current value is kept. (default: None)

            noise_stddev (float or torch.Tensor): Standard deviation for the Gaussian observation noise model. If None,
            current value is kept. (default: None)

            mean_params: Parameters passed on to the mean function. If None, current value is kept. (default: None)
        """
        self._set_lengthscale(lengthscale)
        if signal_stddev is not None:
            self._set_signal_stddev(signal_stddev)
        if noise_stddev is not None:
            self._set_noise_stddev(noise_stddev)
        if mean_params is not None:
            self.mean_function.set_parameters(mean_params)

        if self.Y is None:
            self.clear_data()  # updates internals even if there is no data
        else:
            self.set_data(self.X, self.Y)  # recomputes internals

    def get_hyperparameters(self):
        """
        Method to retrieve GP hyper-parameters.
        
        Returns:
            torch.Tensor or tuple: A single tensor containing all the GP hyper-parameters in the same order as set_
            hyperparameters() or a tuple with all GP hyper-paramters, except the mean parameters, followed by mean
            parameters in separate.
        """
        mean_params = self.mean_function.get_parameters()
        basic_params = torch.stack((self._lengthscale, self.signal_stddev, self.noise_stddev))
        if isinstance(mean_params, torch.Tensor):
            return torch.cat((basic_params, mean_params.view(-1)))
        if mean_params is not None:
            return basic_params, mean_params
        return basic_params

    def clear_data(self):
        """
        Method to clear internal data and reset the model
        """
        # Cholesky factor:
        self.training_mat = self.noise_stddev * torch.eye(self.n_frequencies * 2, dtype=self.dtype, device=self.device)
        # Vector of feature-observation products:
        self.training_vec = torch.zeros((self.n_frequencies * 2, 1), dtype=self.dtype, device=self.device)
        self._update_weights()
        self.X = None
        self.Y = None

    def feature_transform(self, X):
        """
        Method to compute the feature map for a given query point.
        
        Parameters:
            X (torch.Tensor): A N-by-D matrix with N rows of query points, each with D dimensions,
            which should match the model's dimensionality.
            
        Returns:
            torch.Tensor: A M-by-N matrix of Fourier cossine-sine features,
            where M is twice the number of frequencies and N is the number of query points.
        """
        return self.signal_stddev * cos_sin(X, self.spec)

    def updated_training_mat(self, feature):
        """
        Method to compute the updated Cholesky factor of the GP training matrix without performing the internal update,
        useful to estimate information gain.
        
        Parameters:
            feature (torch.Tensor): The feature map of the new data point to include.
            
        Returns:
            torch.Tensor: A M-by-M upper triangular matrix corresponding to the updated Cholesky factor.
        """
        return torch.qr(torch.cat((self.training_mat, feature.t()), dim=0))[1]

    def _update_dataset(self, x, y):
        """
        Method used internally to update the GP dataset with a single data point.
        """
        if self.X is None:
            self.X = x
        else:
            self.X = torch.cat((self.X, x), dim=0)
        if self.Y is None:
            self.Y = y.view(-1, 1)
        else:
            self.Y = torch.cat((self.Y, y.view(-1, 1)), dim=0)

    def update(self, x, y):
        """
        Method to perform incremental update of the GP using a single observation pair.
        
        Parameters:
            x (torch.Tensor): Single data point, formatted as a row vector.
            y (torch.Tensor): Single observation value, formatted as a scalar.
        """
        util.check_exact_dim(x, 2, msg="Data point should be a row vector")
        util.check_exact_dim(y, 0, msg="Observation value should be a scalar")
        phi_t = self.feature_transform(x)
        self.training_vec += phi_t * (y - self.mean_function(x).squeeze())
        self.training_mat = self.updated_training_mat(phi_t)
        self._update_weights()
        self._update_dataset(x, y)

    def _update_weights(self):
        """
        Method internally used to update GP weights' posterior mean
        """
        self.weights_train = torch.cholesky_solve(self.training_vec, self.training_mat,
                                                  upper=True)  # not to be confused with BLR weights sample

    def set_data(self, X_all, Y_all):
        """
        Method to set the GP observations data all at once.
        
        Parameters:
            X_all (torch.Tensor): A N-by-D matrix of D-dimensional query points
            Y_all (torch.Tensor): A N-by-1 matrix of observation values
        """
        util.check_exact_dim(Y_all, 2, msg="Invalid data tensor dimensions. Both X and Y should be matrices.")
        util.check_exact_dim(X_all, 2, msg="Invalid data tensor dimensions. Both X and Y should be matrices.")
        util.check_size(X_all, self.get_dimensionality(), 1, "Input data dimensionality does not match model's setting")

        features = self.feature_transform(X_all)
        self.training_vec = torch.matmul(features, (Y_all - self.mean_function(X_all)))
        mat_A = torch.matmul(features, features.t()) + (self.noise_stddev ** 2) * torch.eye(features.size(0))
        self.training_mat = torch.cholesky(mat_A, upper=True)
        self._update_weights()
        self.X = X_all
        self.Y = Y_all

    def evaluate_nlml(self, lengthscale, signal_stddev=None, noise_stddev=None, mean_params=None, X=None, Y=None,
                      enforce_constraints=False):
        """
        Computes the negative log marginal likelihood (NLML) of the model's hyper-parameters with respect to the data
        
        Parameters:
            lengthscale (float or torch.Tensor): Length-scale of the GP kernel.
            signal_stddev (float or torch.Tensor): Signal standard deviation, i.e. a multiplicative scaling factor for
            the feature maps. If None, current value is used. (default: None)
            noise_stddev (float or torch.Tensor): Standard deviation for the Gaussian observation noise model. If None,
            current value is used. (default: None)
            mean_params: Parameters passed on to the mean function. If None, current value is used. (default: None)
            X (torch.Tensor): Data points to evaluate the NLML on. If None, uses model's dataset. (default: None)
            Y (torch.Tensor): Observation values corresponding to points in X. If None, uses model's dataset. (default:
            None)
            enforce_constraints (bool): Whether or not to enforce positivity constraints on the first three parameters.
            (default: False)

        Returns:
            torch.Tensor: Scalar corresponding to NLML value
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

        n_freqs = self.raw_spec.shape[0]
        n_data = Y.shape[0]
        features = signal_stddev * cos_sin(X, self.raw_spec / lengthscale)
        Y_diff = Y - self.mean_function(X, param=mean_params)
        training_vec = torch.matmul(features, Y_diff)
        mat_A = torch.matmul(features, features.t()) + (noise_stddev ** 2) * torch.eye(features.shape[0])
        weights_train, _ = torch.solve(training_vec, mat_A)
        lml_1_1 = torch.mm(Y_diff.t(), Y_diff)
        lml_1_2 = torch.mm(weights_train.t(), training_vec)
        lml_1 = -(lml_1_1 - lml_1_2) / (2 * noise_stddev ** 2)
        lml_2 = -torch.log(torch.cholesky(mat_A).diagonal()).sum()
        lml = lml_1 + lml_2 + n_freqs * torch.log(noise_stddev) - 0.5 * n_data * torch.log(
            2 * math.pi * noise_stddev ** 2)
        return -lml

    def predict(self, X_test):
        """
        GP inference method.
        
        Parameters:
            X_test (torch.Tensor): A N-by-D matrix of D-dimensional query points
            
        Returns:
            tuple: A tuple of tensors containing the predictions' mean vector followed by the predictions' covariance
            matrix
        """
        phi_test = self.feature_transform(X_test)
        prior_mean = self.mean_function(X_test)
        mean_test = prior_mean + torch.matmul(phi_test.t(), self.weights_train)
        covar_test = (self.noise_stddev ** 2) * torch.matmul(phi_test.t(),
                                                             torch.cholesky_solve(phi_test, self.training_mat,
                                                                                  upper=True))

        return mean_test, covar_test
