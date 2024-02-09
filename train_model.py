import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary

from data_utils import ImageAudioDataset, collate_fn
from diffusion_model import AudioConditionedDiffusionModel


def load_latest_checkpoint(ckpt_base_dir, ckpt_name, start_from_ckpt, device):
    """Loads the latest checkpoint from the given directory.

    Args:
        ckpt_base_dir (str): The directory containing the checkpoints.
        ckpt_name (str): The name of the checkpoint to get the models from.
        start_from_ckpt (bool): Whether to start from the latest checkpoint or from scratch.
        device (str): The device to load the model to.

    Returns:
        tuple: A tuple containing the model and optimizer states and the epoch we start from.
    """
    # find latest epoch in the base directory and then load the model and optimizer
    # from that epoch
    epoch = 0

    model = AudioConditionedDiffusionModel(ckpt_name=ckpt_name)
    if start_from_ckpt and os.path.exists(os.path.join(ckpt_base_dir, "model")):
        model_name = sorted(os.listdir(os.path.join(ckpt_base_dir, "model")))[-1]
        model_ckpt = torch.load(os.path.join(ckpt_base_dir, "model", model_name))
        model.load_state_dict(model_ckpt)
        epoch = int(model_name.split(".")[0].split("_")[1]) + 1

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.get_params(), lr=5e-3, weight_decay=1e-5)
    if start_from_ckpt and os.path.exists(os.path.join(ckpt_base_dir, "optimizer")):
        optimizer_name = sorted(os.listdir(os.path.join(ckpt_base_dir, "optimizer")))[
            -1
        ]
        optimizer_ckpt = torch.load(
            os.path.join(ckpt_base_dir, "optimizer", optimizer_name)
        )
        optimizer.load_state_dict(optimizer_ckpt)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
        threshold_mode="rel",
        min_lr=1e-8,
        threshold=0.01,
    )
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer, gamma=0.8, verbose=True
    # )
    if start_from_ckpt and os.path.exists(os.path.join(ckpt_base_dir, "scheduler")):
        scheduler_name = sorted(os.listdir(os.path.join(ckpt_base_dir, "scheduler")))[
            -1
        ]
        scheduler_ckpt = torch.load(
            os.path.join(ckpt_base_dir, "scheduler", scheduler_name)
        )
        lr_scheduler.load_state_dict(scheduler_ckpt)

    return epoch, model.to(device), optimizer, lr_scheduler


def prepare_ckpt_dir(ckpt_dir):
    if not os.path.exists(os.path.join(ckpt_dir, "model")):
        os.makedirs(os.path.join(ckpt_dir, "model"))
    if not os.path.exists(os.path.join(ckpt_dir, "optimizer")):
        os.makedirs(os.path.join(ckpt_dir, "optimizer"))
    if not os.path.exists(os.path.join(ckpt_dir, "scheduler")):
        os.makedirs(os.path.join(ckpt_dir, "scheduler"))


def main():
    ckpt_name = "stabilityai/stable-diffusion-2"
    batch_size = 8
    num_epochs = 25
    save_every = 5
    ckpt_dir = "checkpoints_10k"
    num_examples = 10_000
    start_from_ckpt = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")

    # Create the dataset
    dataset = ImageAudioDataset(data_dir="data", num_examples=num_examples)
    print(f"dataloader will use {os.cpu_count()} workers")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
    )

    start_epoch, model, optimizer, lr_scheduler = load_latest_checkpoint(
        ckpt_dir, ckpt_name, start_from_ckpt, device
    )
    model.train()

    if start_epoch == 0:
        print("starting from scratch")
    else:
        print(f"starting from result of epoch {start_epoch-1}")

    prepare_ckpt_dir(ckpt_dir)

    train_losses = []

    dummy_batch = next(iter(data_loader))
    summary(
        model,
        input_data=dict(img=dummy_batch[1], aud=dummy_batch[2]),
        device=device,
        mode="train",
    )

    best_loss = sys.maxsize
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(dataset) // batch_size, leave=False
        )
        for i, batch in pbar:
            _, img, aud = batch
            img = img.to(device)
            aud = aud.to(device)

            predicted_noise, target_noise = model(img, aud)
            loss = F.mse_loss(
                predicted_noise.float(), target_noise.float(), reduction="sum"
            )
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": epoch_loss / ((i + 1) * batch_size)})

            loss /= batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(dataset)
        train_losses.append(avg_loss)
        lr_scheduler.step(metrics=avg_loss)
        # lr_scheduler.step()

        print(
            f"Epoch {epoch + 1} average loss: {avg_loss}, learning rate: {optimizer.param_groups[0]['lr']}"
        )

        with open(f"{ckpt_dir}/metrics.txt", "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_loss},{optimizer.param_groups[0]['lr']}\n")

        found_new_best_model = avg_loss < best_loss
        if found_new_best_model:
            best_loss = avg_loss
        if epoch % save_every == 0 or found_new_best_model or epoch == num_epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, "model", f"model_{str(epoch).zfill(4)}.pt"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    ckpt_dir, "optimizer", f"optimizer_{str(epoch).zfill(4)}.pt"
                ),
            )
            torch.save(
                lr_scheduler.state_dict(),
                os.path.join(
                    ckpt_dir, "scheduler", f"scheduler_{str(epoch).zfill(4)}.pt"
                ),
            )


if __name__ == "__main__":
    main()
