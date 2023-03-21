import torchaudio
import torch
from .config import signal_config

def wav2fbank(num_mel_bins):
    """
    Returns a function to convert a waveform to a log Mel-scaled spectrogram using either Kaldi or TorchAudio.

    Args:
        None

    Returns:
        function: A function to convert a waveform to a log Mel-scaled spectrogram.
    """
    def kaldi_fbank(waveform):
        """
        Converts a waveform to a log Mel-scaled spectrogram using Kaldi.

        Args:
            waveform (torch.Tensor): A 1D Tensor representing a waveform.

        Returns:
            torch.Tensor: A 2D Tensor representing a log Mel-scaled spectrogram.
        """
        # Compute the log Mel-scaled spectrogram using Kaldi
        return torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=signal_config['sample_rate'],
            frame_length=signal_config['frame_length'],
            frame_shift=signal_config['frame_shift'],
            htk_compat=True,
            window_type=signal_config['window_type'],
            num_mel_bins=num_mel_bins
        )

    def torchaudio_fbank(waveform):
        """
        Converts a waveform to a log Mel-scaled spectrogram using TorchAudio.

        Args:
            waveform (torch.Tensor): A 1D Tensor representing a waveform.

        Returns:
            torch.Tensor: A 2D Tensor representing a log Mel-scaled spectrogram.
        """
        # Compute the log Mel-scaled spectrogram using TorchAudio
        sample_rate = signal_config['sample_rate']
        frame_length = signal_config['frame_length']
        frame_shift = signal_config['frame_shift']
        window_type = signal_config['window_type']
        sample_num_per_ms = int(sample_rate / 1000)
        frame_length_samples = int(sample_num_per_ms * frame_length)
        frame_shift_samples = int(sample_num_per_ms * frame_shift)

        # Set the window function based on the window_type parameter
        if window_type == 'hanning':
            window_fn = torch.hann_window
        elif window_type == 'hamming':
            window_fn = torch.hamming_window
        elif window_type == 'blackman':
            window_fn = torch.blackman_window

        # Create the MelSpectrogram transform object
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=frame_length_samples,
            hop_length=frame_shift_samples,
            center=False,
            f_min=20,
            power=2.0,
            n_mels=num_mel_bins,
            window_fn=window_fn
        )

        # Compute the log Mel-scaled spectrogram
        return torch.log10(mel_spectrogram(waveform) + 1e-6)

    # Choose the appropriate function based on the compliance parameter in the signal_config dictionary
    if signal_config['compliance'] == 'kaldi':
        return kaldi_fbank
    elif signal_config['compliance'] == 'torchaudio':
        return torchaudio_fbank


def fbank_based_vad(fbank, energy_threshold, bin_threshold, mask_threshold):
    # Compute the energy of each frame
    energy_map = fbank > energy_threshold
    frame_map = energy_map.sum(dim=1)

    # Compute the binary mask
    mask = frame_map > bin_threshold

    # Find the starting index to crop
    start_idx = 0
    for i, val in enumerate(mask):
        if val:
            if i + mask_threshold > len(mask) or all(mask[i:i + mask_threshold]):
                start_idx = i
                break

    # Find the ending index to crop
    end_idx = len(mask)
    for i, val in reversed(list(enumerate(mask))):
        if val:
            if i - mask_threshold + 1 < 0 or all(mask[i - mask_threshold + 1:i + 1]):
                end_idx = i + 1
                break

    # Crop fbank according to the mask
    cropped_fbank = fbank[start_idx:end_idx]

    return cropped_fbank