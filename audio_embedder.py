import torch
import torch.nn as nn
from torchaudio.transforms import Resample
from transformers import AutoModel


class AudioEmbedder(nn.Module):
    """Produces an audio representation from a raw waveform."""

    def __init__(self, embedding_size: int = 768, trainable_base: bool = False):
        super().__init__()

        hidden_size = 768

        self.resample = Resample(orig_freq=44100, new_freq=16000)
        self.wav2vec = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
        if not trainable_base:
            self.wav2vec.freeze_feature_encoder()
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, embedding_size)
        self.activation = nn.GELU()

    def forward(self, x):
        """Defines the forward pass of the audio encoder.

        Args:
            x (torch.Tensor): Raw audio waveform tensor of size (batch_size, seq_len).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len, embedding_size) containing the
                audio representation.
        """
        # Resample to 16kHz expected by Wav2Vec2
        x = self.resample(x)
        # Normalize to [-1, 1] as expected by Wav2Vec2
        x /= torch.max(torch.abs(x))
        # run wav2vec2 and aggregate the output along temporal dimension
        x = self.wav2vec(x, output_hidden_states=True).last_hidden_state
        # 2 linear projections with a gelu activation in between
        x = self.activation(self.proj1(x))
        return self.proj2(x)
