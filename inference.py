import os

import torch
import tqdm
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from data_utils import ImageAudioDataset, collate_fn
from diffusion_model import AudioConditionedDiffusionModel
from train_model import load_latest_checkpoint


def main():
    print("Starting main function")
    checkpoint_dir = "scratch/ds7337/code_ori/checkpoints_20k_768"
    batch_size = 1
    num_examples = 8
    output_dir = "denoised_images"
    split = "train"

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the dataset
    dataset = ImageAudioDataset(
        data_dir="/scratch/ds7337/small_data",
        num_examples=num_examples,
        split=split,
    )
    print(f"dataloader will use {os.cpu_count()} workers")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
    )

    _, model, _, _ = load_latest_checkpoint(
        checkpoint_dir,
        "CompVis/stable-diffusion-v1-4",
        True,
        device,
    )
    #    model = AudioConditionedDiffusionModel("stabilityai/stable-diffusion-2")
    model.eval()
    scheduler = LMSDiscreteScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )
    vae = model.vae.to(device)
    tok_and_encode = model.audio_embedder.to(device)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    ).to(device)
    #   scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    num_inference_steps = 100  # Number of denoising steps
    guidance_scale = 1.5  # Scale for classifier-free guidance
    batch_size = 2
    ctr = 0
    with torch.no_grad():
        for _, (names, _, aud) in tqdm(
            enumerate(data_loader), total=len(dataset) // batch_size
        ):
            text_embeddings = tok_and_encode(aud.cuda()).to(device)
            #            print(text_embeddings.shape)
            latents = torch.randn(1, 4, 64, 64).to(device)
            latents = latents.to(device)
            #            print(latents.shape)

            scheduler.set_timesteps(num_inference_steps)

            latents = latents * scheduler.init_noise_sigma

            for t in tqdm(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                #                latent_model_input = torch.cat([latents] * 2)
                # print(latents.shape, latent_model_input.shape)

                latent_model_input = scheduler.scale_model_input(latents, t)
                #                print("LM SHAPE: ", latent_model_input.shape)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

                # perform guidance
                #               print("NOISE PRED: ", noise_pred.shape)
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #                noise_pred = (noise_pred + guidance_scale * (noise_pred/2))

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = 1 / 0.18215 * latents

            with torch.no_grad():
                image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            for img in pil_images:
                img = img.save(str(names) + "_20k678.jpg")
                ctr += 1


if __name__ == "__main__":
    main()
