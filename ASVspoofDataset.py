import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample


class ASVspoofDataset(Dataset):
    def __init__(self, data_path, protocol_file, transform=None):
        self.data_path = data_path
        self.transform = transform

        # Read protocol file and store the file paths and labels
        self.audio_files = []
        self.labels = []
        with open(protocol_file, "r") as f:
            for line in f.readlines():
                items = line.strip().split()
                audio_file = items[1]
                label = 1 if items[4] == "spoof" else 0  # 1 for spoof, 0 for bonafide
                self.audio_files.append(audio_file)
                self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # Load the audio file
        audio_file_path = os.path.join(self.data_path, self.audio_files[index] + ".flac")
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # Resample the audio waveform to 16kHz if needed
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Apply transformations if any
        if self.transform:
            waveform = self.transform(waveform)

        # Get the label
        label = self.labels[index]

        return waveform, label
