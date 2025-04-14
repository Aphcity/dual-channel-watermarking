import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
print(torch.__version__)

print(torch.cuda.is_available())  # This should return True
print(torch.version.cuda)

# optimize_for_mobile()