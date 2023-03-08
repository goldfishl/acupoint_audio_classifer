import torchaudio
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os

from src.utils import acup_config, load_split



class AudioDataset(Dataset):
    def __init__(self, model_config, data_config):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        
        self.data, self.label = load_split(data_config['data_path'], data_config['split_file'])
        
        self.freqm = data_config['freq_mask']
        self.timem = data_config['time_mask']
        self.mixup = data_config['mixup']
        self.norm_mean = data_config['norm_mean']
        self.norm_std = data_config['norm_std']
        self.skip_norm = data_config['skip_norm']
        self.noise = data_config['noise']
        self.wav2fbank = data_config['feature']
        self.target_length = model_config['target_length']
        self.data_path = data_config['data_path']

        with open(data_config['label_file'], "r") as file:
            lines = file.readlines()
            label = [line.strip() for line in lines]
        self.label_to_index = {label[i]: i for i in range(len(label))}
        self.label_num = model_config['num_classes']
    
    def maxup_process(self, waveform, label):
        # find another sample to mix
        mix_sample_idx = random.randint(0, len(self.data)-1)
        mix_data = self.data[mix_sample_idx]
        mix_waveform, _ = torchaudio.load(os.path.join(self.data_path, mix_data))
        mix_waveform = mix_waveform - mix_waveform.mean()

        # align the length of two waveforms for mixup
        if waveform.shape[1] != mix_waveform.shape[1]:
            if waveform.shape[1] > mix_waveform.shape[1]:
                # padding
                temp_wav = torch.zeros(1, waveform.shape[1])
                temp_wav[0, 0:mix_waveform.shape[1]] = mix_waveform
                mix_waveform = temp_wav
            else:
                # cutting
                mix_waveform = mix_waveform[0, 0:waveform.shape[1]]

        mix_lambda = np.random.beta(10, 10)

        mixed_waveform = mix_lambda * waveform + (1 - mix_lambda) * mix_waveform
        mixed_waveform = mixed_waveform - mixed_waveform.mean()

        fbank = self.wav2fbank(mixed_waveform)

        label_indices =  torch.zeros(self.label_num)
        label_indices[self.label_to_index[label]] = mix_lambda
        label_indices[self.label_to_index[self.label[mix_sample_idx]]] = 1 - mix_lambda
        label_indices = torch.FloatTensor(label_indices)

        return fbank, label_indices



    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        data = self.data[index]
        waveform, _ = torchaudio.load(os.path.join(self.data_path, data))
        waveform = waveform - waveform.mean()
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            fbank, label_indices = self.maxup_process(waveform, self.label[index]) 
            
        # if not do mixup
        else:
            fbank = self.wav2fbank(waveform)
            label_indices =  torch.zeros(self.label_num)
            label_indices[self.label_to_index[self.label[index]]] = 1.0


        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        # add noise to the input
        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)     

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0, 1).unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze().transpose(0, 1)
        
        # cut and pad the input to the target length
        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        
        return fbank, label_indices

    def __len__(self):
        return len(self.data)