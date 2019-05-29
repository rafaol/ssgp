import torch

def ensure_torch(value,dtype=torch.float32):
    if not isinstance(value,torch.Tensor):
        value = torch.as_tensor(value,dtype=dtype)
    return value

def ensure_positive(v):
    return torch.nn.functional.softplus(v)

