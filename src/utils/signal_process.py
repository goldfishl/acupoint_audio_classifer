import torchaudio
import torch


def wav2fbank(num_mel_bins, compliance='kaldi'):
    def kaldi_fbank(waveform):
        # default sample_frequency:16000.0, frame_length:25(ms), frame_shift:10(ms), use_energy = False
        return torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, window_type='hanning', 
                                                 num_mel_bins=num_mel_bins)
    def torchaudio_fbank(waveform):
        # default sample_rate:16000, n_fft:400[16 * 25ms], win_length:n_fft, mel_scale='htk'
        _fank = torchaudio.transforms.MelSpectrogram(
                            n_fft=400,
                            hop_length=160,
                            center=False,
                            f_min=20,
                            power=2.0,
                            n_mels=num_mel_bins,
                            # norm="slaney",
                        )
        return torch.log10(_fank(waveform)+1e-6)

    if compliance == 'kaldi':
        return kaldi_fbank
    elif compliance == 'torchaudio':
        return torchaudio_fbank