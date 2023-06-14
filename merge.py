import torchaudio
import torch

a = torchaudio.load('LA_T_1000648.flac')
b = torchaudio.load('LA_T_9995976.flac')

# let a[0] and b[0] be two tensors of shape (1, 16000)
print(len(torch.split(a[0],30000,dim= 1)))
print(a[1])



# a + b

c = 0.4*torch.split(a[0],30000,dim= 1)[0] + 0.6*torch.split(b[0],30000,dim= 1)[0]
torchaudio.save('c.flac',c,a[1]) 