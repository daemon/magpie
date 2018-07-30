import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):

    def __init__(self, filter_length=480, hop_length=160):
        super().__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def forward(self, input_data):
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
        num_samples = input_data.size(1)
        input_data = input_data.unsqueeze(1)
        forward_transform = F.conv1d(input_data, self.forward_basis, stride=self.hop_length, 
            padding=self.filter_length)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        return magnitude


class MelSpectrogram(nn.Module):

    def __init__(self, sample_rate=16000, filter_length=480, hop_length=160, num_mels=80):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate  
        
        self.stft = STFT(filter_length=self.filter_length, hop_length=self.hop_length)
        mel_filters = librosa.filters.mel(self.sample_rate, self.filter_length, self.num_mels)
        self.register_buffer("mel_filter_bank", torch.FloatTensor(mel_filters))

    def forward(self, input_data):
        magnitude = self.stft(input_data)
        mel_spectrogram = F.linear(magnitude.transpose(-1, -2), self.mel_filter_bank)
        mel_spectrogram = 10.0 * (torch.log(mel_spectrogram**2 + 1e-8) / np.log(10.0))
        return mel_spectrogram
