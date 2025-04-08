import torch

DTYPES = [torch.float32, torch.float64]

DEVICES = (
    [torch.device("cpu"), torch.device("cuda")]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)
