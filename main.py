import efficientv2
import torch


DEVICE: str = torch.device("cuda:"+ str(GPU) if torch.cuda.is_available() else "cpu")
# train efficientv2

model = efficientv2.effnetv2_m()
model.to(DEVICE)


