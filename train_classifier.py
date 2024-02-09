import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from classifier_data_utils import ImageAudioDataset, collate_fn
from diffusion_model import AudioConditionedDiffusionModel

device = torch.device("cuda")


model = AudioConditionedDiffusionModel(ckpt_name="stabilityai/stable-diffusion-2")
model.load_state_dict(torch.load("model_0024.pt"))
model_diff = model.to(device)


model = nn.Sequential(
    nn.Conv2d(1280, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(256, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(output_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(128 * 4, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

model = model.to(device)


def hook_fn(module, input, output):
    module.output = output


model_diff.unet.mid_block.register_forward_hook(hook_fn)


dataset = ImageAudioDataset(data_dir="/scratch/ds7337/data", num_examples=26216)
print(f"dataloader will use {os.cpu_count()} workers")
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
)

print(len(dataset))
# In[8]:


# dummy_batch = next(iter(data_loader))
# # summary(
# #     model_diff,
# #     input_data=dict(img=dummy_batch[1], aud=dummy_batch[2]),
# #     device="cpu",
# #     mode="train",
# # )


# In[9]:


batch_size = 8


# In[10]:


loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=2
)


# In[13]:


train_losses = []
best_loss = sys.maxsize
prev_loss = float("inf")

for epoch in range(4):
    epoch_loss = 0
    pbar = tqdm.tqdm(
        enumerate(data_loader), total=len(dataset) // batch_size, leave=False
    )
    for i, batch in pbar:
        _, img, aud, labels = batch
        if img is None or aud is None or labels is None:
            continue
        img = img.to(device)
        aud = aud.to(device)
        labels = labels.to(device)
        # Get middle block from model
        predicted_noise, target_noise = model_diff(img, aud)
        x = model_diff.unet.mid_block.output
        x = (x - torch.mean(x)) / torch.std(x)
        # print("UNet range: ", min(x.flatten()), max(x.flatten()), (x.flatten().mean()))
        x = model(x.cuda())
        # print("OP range: ", min(x.flatten()), max(x.flatten()), (x.flatten().mean()))
        x = F.softmax(x.squeeze(), dim=-1)
        # print("Softmax range: ", min(x.flatten()), max(x.flatten()), (x.flatten().mean()))
        if x.dim() == 0:
            continue
        loss = loss_fn(x, labels)
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": epoch_loss / ((i + 1) * batch_size)})
        loss /= batch_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    #        del x
    avg_loss = epoch_loss / len(dataset)
    if avg_loss < prev_loss:
        torch.save(model.state_dict(), "classifier_" + str(epoch) + ".pt")
        prev_loss = avg_loss
        print("Best model saved")

    train_losses.append(avg_loss)
    scheduler.step(metrics=avg_loss)

with open("loss_arr.pkl", "wb") as f:
    pickle.dump(train_losses, f)

print("Losses saved")


# In[12]:


torch.save(model.state_dict(), "classifier_30k.pt")
print("Model Saved")
