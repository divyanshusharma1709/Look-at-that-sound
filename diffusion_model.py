import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from audio_embedder import AudioEmbedder


class AudioConditionedDiffusionModel(torch.nn.Module):
    """A diffusion model conditioned on audio data."""

    def __init__(
        self, ckpt_name, freeze_vae=True, freeze_unet=True, freeze_wav2vec=True
    ):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(ckpt_name, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(ckpt_name, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            ckpt_name, subfolder="scheduler"
        )
        self.audio_embedder = AudioEmbedder(
            embedding_size=1024, trainable_base=not freeze_wav2vec
        )

        self.freeze_vae = freeze_vae
        self.freeze_unet = freeze_unet
        if self.freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
        if self.freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False

    def get_params(self):
        """Returns the parameters of the model. If freeze_vae or freeze_unet is True, the
        parameters of the respective model are not returned.

        Returns:
            list: List of model parameters to update during training.
        """
        params = []
        if not self.freeze_vae:
            params.extend(self.vae.parameters())
        if not self.freeze_unet:
            params.extend(self.unet.parameters())
        params.extend(self.audio_embedder.parameters())
        return params

    def forward(self, img=None, aud=None, total_denoising_steps=None):
        """The forward pass of the model.

        Args:
            img (torch.Tensor): The original image (to noise and denoise)
            aud (torch.Tensor): The audio waveform to condition with.

        Returns:
            torch.Tensor: The loss of the model on the given batch.
        """
        audio_emb = self.audio_embedder(aud)

        if self.training:
            assert img is not None and aud is not None

            batch_size = img.shape[0]
            latent_dist = self.vae.encode(img).latent_dist
            latents = latent_dist.sample()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device,
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            predicted_noise = self.unet(noisy_latents, timesteps, audio_emb).sample
            return predicted_noise, noise
        else:
            assert (
                img is not None
                and aud is not None
                and total_denoising_steps is not None
            )
            for denoising_steps in reversed(range(total_denoising_steps)):
                predicted_noise = self.unet(img, denoising_steps, audio_emb).sample
                img -= predicted_noise
            return img - predicted_noise
