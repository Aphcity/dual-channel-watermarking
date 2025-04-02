import torch
print(torch.__version__)

print(torch.cuda.is_available())  # This should return True
print(torch.version.cuda)