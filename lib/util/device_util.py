import torch

cpu = torch.device('cpu')
cuda = torch.device('cuda')

model_device = cuda
minimal_dta_device = cpu
drug_device = cuda
target_device = cpu

dtype = torch.float
