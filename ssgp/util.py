import torch


def ensure_torch(value, dtype=torch.float32, device=None):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=dtype, device=device)
    return value.to(device=device, dtype=dtype)


def ensure_positive(v):
    return torch.nn.functional.softplus(v)


def check_exact_dim(x, dim, msg=None):
    if not torch.is_tensor(x) or x.dim() != dim:
        if msg is None:
            msg = "Expected tensor of dimensionality {}".format(dim)
        raise ValueError(msg)


def check_size(x, size, dim=0, msg=None):
    if not x.size(dim) == size:
        if msg is None:
            msg = "Expected tensor size at {} to be {}".format(dim, size)
        raise ValueError(msg)
