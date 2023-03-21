import torchaudio
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
from src.utils import fbank_based_vad, load_split, load_label
from .config import exp_config



class AudioDataset(Dataset):
    def __init__(self, model_config, data_config):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        
        self.data, self.label = load_split(data_config)
        
        self.freqm = data_config['freq_mask']
        self.timem = data_config['time_mask']
        self.norm_mean = data_config['norm_mean']
        self.norm_std = data_config['norm_std']
        self.skip_norm = data_config['skip_norm']
        self.noise = data_config['noise']
        self.wav2fbank = data_config['feature']
        self.target_length = model_config['target_length']
        self.data_path = data_config['data_path']

        self.vad = data_config['vad']
        self.energy_threshold = data_config['vad_energy_threshold']
        self.bin_threshold = data_config['vad_bin_threshold']
        self.mask_threshold = data_config['vad_mask_threshold']

        label = load_label(data_config['label_file'])
        self.label_to_index = {label[i]: i for i in range(len(label))}
        self.label_num = model_config['num_classes']
    

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
            

        fbank = self.wav2fbank(waveform)
        label = self.label_to_index[self.label[index]]
        label = torch.tensor(label)


        # # fbank based VAD
        # if self.vad:
        #     fbank = fbank_based_vad(fbank, self.energy_threshold,
        #                             self.bin_threshold, self.mask_threshold)

        # add noise to the input
        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

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
        
        fbank = fbank.transpose(0, 1)
        return fbank, label

    def __len__(self):
        return len(self.data)