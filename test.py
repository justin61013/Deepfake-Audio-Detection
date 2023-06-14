import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ASVspoofDataset import ASVspoofDataset, ASVspoofDataset_mix, AudioDataset
from net import Model1, SoundStream, Model2, Model3
from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

C = 1
D = 1
hidden_units = 128
learning_rate = 0.0005
batch_size = 32
num_epochs = 180
max_length = 80000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_model = Model2(C, D, hidden_units).to(device)

eval_data_path = r"dataset\ASVspoof\test\audio"
eval_protocol_file = r"dataset\ASVspoof\ASVspoof2019.LA.cm.eval.trl.txt"
eval_dataset = AudioDataset(eval_data_path)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)



# Evaluate on development (dev) set
new_model.load_state_dict(torch.load("weight/trained_model4.pth"))
new_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in eval_loader:
        data, targets = data.to(device), targets.to(device)
        scores = new_model(data)
        predictions = torch.argmax(scores, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    accuracy = correct / total
    print(f"Development Set Accuracy: {accuracy:.4f}")

#  calculate new_model parameters and FLOPs
# from thop import profile
# from thop import clever_format

# input = torch.randn(1, 1, 80000).to(device)
# macs, params = profile(new_model, inputs=(input, ))
# macs, params = clever_format([macs, params], "%.3f")
# print(macs, params)
