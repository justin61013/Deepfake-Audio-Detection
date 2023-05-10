import torchaudio
import torch

a = torchaudio.load('LA_T_1001169.flac')
b = torchaudio.load('LA_T_1000648.flac')

# let a[0] and b[0] be two tensors of shape (1, 16000)
print(len(torch.split(a[0],30000,dim= 1)))
print(a[1])



# a + b

c = torch.split(a[0],30000,dim= 1)[0] + torch.split(b[0],30000,dim= 1)[0]
torchaudio.save('c.flac',c,a[1]) 