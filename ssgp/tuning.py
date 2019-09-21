# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:11:12 2019

@author: Rafael Oliveira
"""

import torch

class NLoptTuningObjective:
    """
    Class implementing objective function to be used with NLopt to tune SSGP hyper-parameters based on the negative log-marginal likelihood (NLML).
    """
    def __init__(self,model,param_names=["lengthscale"],compute_grad=False):
        """
        Constructor.
        
        Parameters:
            model (ssgp.models.ISSGPR): Model to tune
            param_names (list): List of parameter names matching ssgp.models.ISSGPR.evaluate_nlml() argument names (default: ['lengthscale'])
            compute_grad (bool): Whether or not to compute NLML gradients, which uses pytorch's autograd (default: False)
        """
        self.param_names = param_names
        self.model = model
        self.compute_grad = compute_grad
        
    def hp_map(self,x,compute_grad=False):
        """
        Maps tensor to dictionary of hyper-parameters
        
        Parameters:
            x (iterable): Array of hyper-parameter values
            compute_grad(bool): Whether or not to require gradients computation. (default: False)
        
        Returns:
            dict: Dictionary indexed by hyper-parameter names passed to constructor and values from input argument.
        """
        param_dict = dict.fromkeys(self.param_names)
        for i,p in enumerate(self.param_names):
            param_dict[p] = self.model.ensure_torch(x[i]).requires_grad_(compute_grad)
        return param_dict

    def __call__(self,x,grad):
        """
        Evaluates model's NLML and computes gradients, if enabled.
        
        Parameters:
            x (numpy.ndarray): Array of hyper-parameter values
            grad (numpy.ndarray): Gradient array to be updated
        
        Returns:
            float: NLML value
        """
        param_dict = self.hp_map(x,self.compute_grad)
        loss = self.model.evaluate_nlml(**param_dict)
        if self.compute_grad:
            loss.backward()
            grads = [param_dict[p].grad for p in param_dict]
            grad[:] = torch.stack(grads).cpu().numpy()
        return loss.item()