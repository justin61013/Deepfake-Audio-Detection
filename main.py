import efficientv2
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# train efficientv2

model = efficientv2.effnetv2_m()
model.to(DEVICE)


