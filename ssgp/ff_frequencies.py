# Anthony's code to generate frequencies for random Fourier features of different kernels whose inverse cumulative distribution functions are analytic
import math
import torch
import torch.distributions as tdists


def normal(points, loc, scale):
    """
    Standard Normal
    :param points:
    :param loc:     mean
    :param scale:   standard deviation
    :return:
    """
    _loc = torch.tensor(loc)
    _scale = torch.tensor(scale)
    p = tdists.Normal(loc=_loc, scale=_scale)
    return p.icdf(points)


def standard_normal(points):
    """
    Standard Normal
    :param points:
    :return:
    """
    _loc = torch.tensor(0.0)
    _scale = torch.tensor(1.0)
    p = tdists.Normal(loc=_loc, scale=_scale)
    return p.icdf(points)


def matern_12(points):
    """
    Matern 12
    Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :return:
    """
    return torch.tan(math.pi * (points - 0.5))


def matern_32(points):
    """
    Matern 32
    ref: Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :return:
    """
    return (2 * points - 1) / torch.sqrt(2 * points * (1 - points))


def matern_52(points):
    """
    Matern 52
    Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :return:
    """
    alpha = 4 * points * (1 - points)
    p = 4 / torch.sqrt(alpha) * torch.cos((1 / 3) * torch.acos(torch.sqrt(alpha)))
    return torch.sign(points - 0.5) * torch.sqrt(p - 4)
