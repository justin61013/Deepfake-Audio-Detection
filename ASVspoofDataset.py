import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random
import glob
from moviepy.editor import AudioFileClip
import librosa


class ASVspoofDataset(Dataset):
    def __init__(self, data_path, protocol_file, max_length, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.max_length = max_length

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

        current_length = waveform.size(1)
        if current_length < self.max_length:
            pad_amount = self.max_length - current_length
            waveform = F.pad(waveform, (0, pad_amount))
        elif current_length > self.max_length:
            waveform = waveform[:, :self.max_length]
        # Apply transformations if any
        if self.transform:
            waveform = self.transform(waveform)

        # Get the label
        label = self.labels[index]

        return waveform, label
    



class ASVspoofDataset_mix(Dataset):
    def __init__(self, data_path, protocol_file, max_length, additional_data_path=None, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.max_length = max_length

        # 分别存储label为0和label为1的文件路径和对应的标签
        self.audio_files = []
        self.labels = []

        with open(protocol_file, "r") as f:
            for line in f.readlines():
                items = line.strip().split()
                audio_file = items[1]
                label = 1 if items[4] == "spoof" else 0
                self.audio_files.append(audio_file)
                self.labels.append(label)
                if label == 0:
                    self.audio_files_label0 = self.audio_files.copy()
        if additional_data_path is not None:
            additional_audio_files = glob.glob(os.path.join(additional_data_path, "*.flac"))
            for audio_file in additional_audio_files:
                self.audio_files.append(os.path.splitext(os.path.basename(audio_file))[0])
                self.labels.append(0)
                self.original_labels.append(0)
        self.original_labels = self.labels.copy()



    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file_path = os.path.join(self.data_path, self.audio_files[index] + ".flac")
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # 音频重采样
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # 音频裁剪或填充
        waveform = self._trim_or_pad(waveform)

        # 获取标签
        label = self.labels[index]

        # 如果label为1，则随机混合一个label为0的数据
        if label == 1:
            random_label0_file = random.choice(self.audio_files_label0)
            label0_audio_file_path = os.path.join(self.data_path, random_label0_file + ".flac")
            label0_waveform, sample_rate = torchaudio.load(label0_audio_file_path)

            if sample_rate != 16000:
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                label0_waveform = resampler(label0_waveform)

            label0_waveform = self._trim_or_pad(label0_waveform)

            A = random.uniform(0.6, 1)
            waveform = A * waveform + (1 - A) * label0_waveform
            label = torch.tensor([1 - A, A], dtype=torch.float)
        else:
            label = torch.tensor([1, 0], dtype=torch.float)

        return waveform, label, self.original_labels[index]

    def _trim_or_pad(self, waveform):
        current_length = waveform.size(1)
        if current_length < self.max_length:
            pad_amount = self.max_length - current_length
            waveform = F.pad(waveform, (0, pad_amount))
        elif current_length > self.max_length:
            waveform = waveform[:, :self.max_length]
        return waveform



class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.file_list = [f for f in os.listdir(directory) if f.endswith(".mp4")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video_file = self.file_list[idx]
        video_path = os.path.join(self.directory, video_file)

        # Extract audio from video
        audio = AudioFileClip(video_path)
        audio_path = video_path.split('.')[0] + '.wav'
        audio.write_audiofile(audio_path)
        
        # Load audio as a waveform `y` and sampling rate `sr`
        y, sr = librosa.load(audio_path)

        # Assign label based on file name
        if 'real' in video_file:
            label = 0
        elif 'fake' in video_file:
            label = 1
        else:
            label = -1  # assign a default label or throw an error

        if self.transform:
            y = self.transform(y)

        return y, sr, label

