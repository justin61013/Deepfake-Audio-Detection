import matplotlib.pyplot as plt

model_name = "model8"

train_losses = [0.1,0.2,0.3,0.4,0.5]
eval_losses = [0.2,0.3,0.4,0.5,0.6]
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Curves')
plt.legend()

plt.savefig('loss_curves/'+model_name+'.png', dpi=300)
