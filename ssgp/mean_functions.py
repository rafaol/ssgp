## @package mean_functions
# Mean functions for Gaussian process regression

from abc import ABC, abstractmethod
import torch
from . import util

## Abstract base class for mean functions
class AbstractMeanFunction(ABC):
    ## The constructor initializes the parameters
    def __init__(self,param=None):
        self.set_parameters(param)

    ## Method to set the mean function parameters
    # @param parameters Mean function parameters
    def set_parameters(self,parameters):
        self.parameters = parameters

    ## Retrieve mean function parameters
    def get_parameters(self):
        return self.parameters

    ## Evaluate mean function at x
    # @param x Query point
    @abstractmethod
    def __call__(self,x):
        pass


class ZeroMean(AbstractMeanFunction):
    def __init__(self):
       super().__init__()

    def __call__(self,x,param=None):
        return torch.zeros(x.shape[0],1)


class ConstantMean(AbstractMeanFunction):
    def __init__(self,constant):
        super().__init__(constant)

    def set_parameters(self,param):
        """
        Sets the constant value for the mean function
        """
        param = util.ensure_torch(param)
        if param.numel() > 1 or param.dim()>0:
            raise ValueError("Expected single-element tensor, but got: {}".format(param.size()))
        super().set_parameters(param)

    def __call__(self,x,param=None):
        if param is None:
            constant = self.get_parameters()
        else:
            constant = param
        return torch.ones(x.shape[0],1)*constant
