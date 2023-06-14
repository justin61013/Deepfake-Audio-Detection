import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ASVspoofDataset import ASVspoofDataset, ASVspoofDataset_mix
from net import Model1, SoundStream, Model2, Model3
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Hyperparameters
C = 1
D = 1
hidden_units = 128
learning_rate = 0.0002
batch_size = 32
num_epochs = 180
max_length = 80000

model_name = "model8"
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
class_weights = torch.tensor([1.5, 1.0]).to(device)


# Load dataset
train_data_path = r"dataset\ASVspoof\train\audio"
train_protocol_file = r'dataset\ASVspoof\ASVspoof2019.LA.cm.train.trn.txt'
train_dataset = ASVspoofDataset_mix(train_data_path, train_protocol_file, max_length)

dev_data_path = r"dataset\ASVspoof\valid\audio"
dev_protocol_file = r"dataset\ASVspoof\ASVspoof2019.LA.cm.dev.trl.txt"
dev_dataset = ASVspoofDataset_mix(dev_data_path, dev_protocol_file, max_length)

eval_data_path = r"dataset\ASVspoof\test\audio"
eval_protocol_file = r"dataset\ASVspoof\ASVspoof2019.LA.cm.eval.trl.txt"
eval_dataset = ASVspoofDataset_mix(eval_data_path, eval_protocol_file, max_length)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained model
n_q = 1  # Set the number of quantizers
codebook_size = 1  # Set the codebook size
pretrained_model = SoundStream(C, D, n_q, codebook_size)
pretrained_model.load_state_dict(torch.load("last_model2.pt"))


# Initialize model, criterion, and optimizer
new_model = Model2(C, D, hidden_units).to(device)
# new_model = Model3(C, D,n_q, codebook_size, hidden_units).to(device)
# Transfer the encoder weights from the pre-trained model to the new model
new_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
# new_model.quantizer.load_state_dict(pretrained_model.quantizer.state_dict())

# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.BCEWithLogitsLoss(weight=class_weights)


optimizer = optim.Adam(new_model.parameters(), lr=learning_rate)

train_losses = []
eval_losses = []


# Training loop
for epoch in tqdm(range(num_epochs)):
    new_model.train()
    running_loss = 0
    for batch_idx, (data, targets, original_targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        logits  = new_model(data)

        loss = criterion(logits, targets)
        running_loss += loss.item() * data.size(0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Evaluate on evaluation set
    new_model.eval()
    running_eval_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        fake_correct = 0
        real_correct = 0
        soft_label = 0
        fake_false = 0
        real_false = 0
        for data, soft_targets, original_targets in dev_loader:
            data, soft_targets = data.to(device), soft_targets.to(device)
            scores = new_model(data)
            predictions = torch.argmax(scores, dim=1)
            original_targets = original_targets.to(device)
            correct += (predictions == original_targets).sum().item()
            total += original_targets.size(0)

            for pred, target in zip(predictions, original_targets):
                if pred == target:
                    if target == 1:  # 假设 1 表示 fake
                        fake_correct += 1
                    elif target ==0:  # 假设 0 表示 real
                        real_correct += 1
                    else:
                        soft_label += 1
                else:
                    if target == 1:  # 假设 1 表示 fake
                        fake_false += 1
                    elif target ==0:  # 假设 0 表示 real
                        real_false += 1
                    else:
                        soft_label += 1
                loss = criterion(scores, soft_targets)

                running_eval_loss += loss.item() * data.size(0)
    
            eval_loss = running_eval_loss / len(dev_loader.dataset)
            eval_losses.append(eval_loss)

        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Evaluation Accuracy: {accuracy:.4f}")
        print(f"fake_correct: {fake_correct}, real_correct: {real_correct}, fake_false: {fake_false}, real_false: {real_false}")

torch.save(new_model.state_dict(), "weight/trained_"+model_name+".pth")


plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Curves')
plt.legend()

# Save the figure
plt.savefig('loss_curves/'+model_name+'.png', dpi=300)

# Evaluate on development (dev) set
new_model.load_state_dict(torch.load("weight/trained_"+model_name+".pth"))
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

    

