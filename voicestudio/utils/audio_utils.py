import torch
import torchaudio

import matplotlib.pyplot as plt
from IPython.display import Audio
from librosa import display as rosa_display


def show_waveform(audio_path: str | None, waveform: torch.Tensor | None = None, sr: int = 48000):
    if audio_path:
        waveform, sr = torchaudio.load(audio_path)
    elif waveform is not None:
        waveform = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
    else:
        raise ValueError("Either audio_path or waveform must be provided.")

    plt.figure(figsize=(10, 4))
    rosa_display.waveshow(waveform.numpy()[0], sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    return Audio(waveform.numpy()[0], rate=sr)
