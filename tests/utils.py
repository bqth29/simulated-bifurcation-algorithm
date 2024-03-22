import torch

INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
FLOAT_DTYPES = [torch.float32, torch.float64]
DTYPES = FLOAT_DTYPES + INT_DTYPES

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
