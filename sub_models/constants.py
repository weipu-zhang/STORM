import torch

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
# DEVICE = torch.device("cpu")
DTYPE_16 = torch.float16 if DEVICE.type == "mps" else torch.bfloat16
