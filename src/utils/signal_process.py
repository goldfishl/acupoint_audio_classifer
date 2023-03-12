import torchaudio
import torch
from .config import signal_config



def wav2fbank():
    def kaldi_fbank(waveform):
        # default sample_frequency:16000.0, frame_length:25(ms), frame_shift:10(ms), use_energy = False
        return torchaudio.compliance.kaldi.fbank(waveform, sample_frequency = signal_config['sample_rate'],
                                                 frame_length = signal_config['frame_length'],
                                                    frame_shift = signal_config['frame_shift'],
                                                 htk_compat=True, window_type = signal_config['window_type'],
                                                 num_mel_bins = signal_config['num_mel_bins'])
    def torchaudio_fbank(waveform):
        sample_num_per_ms = int(signal_config['sample_rate'] / 1000)
        frame_length_samples = int(sample_num_per_ms * signal_config['frame_length'])
        frame_shift_samples = int(sample_num_per_ms * signal_config['frame_shift'])

        # torch.hann_window, torch.hamming_window, torch.blackman_window
        if signal_config['window_type'] == 'hanning':
            window = torch.hann_window
        elif signal_config['window_type'] == 'hamming':
            window = torch.hamming_window
        elif signal_config['window_type'] == 'blackman':
            window = torch.blackman_window

        # default sample_rate:16000, n_fft:400[16 * 25ms], win_length:n_fft, mel_scale='htk'
        _fank = torchaudio.transforms.MelSpectrogram(
                            n_fft = frame_length_samples,
                            hop_length = frame_shift_samples,
                            center = False,
                            f_min = 20,
                            power = 2.0,
                            n_mels = signal_config['num_mel_bins'],
                            window_fn = window,
                            # norm="slaney",
                        )
        return torch.log10(_fank(waveform)+1e-6)

    if signal_config['compliance'] == 'kaldi':
        return kaldi_fbank
    elif signal_config['compliance'] == 'torchaudio':
        return torchaudio_fbank