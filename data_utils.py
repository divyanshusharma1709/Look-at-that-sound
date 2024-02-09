import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
# from torchvision import transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224


class ImageAudioDataset(torch.utils.data.Dataset):
    """A dataset class for the image-audio pairs."""

    def _get_split_ids(self, data_dir: str, split: str) -> np.ndarray:
        # get the ids of the examples in the right split
        all_examples = pd.read_csv(os.path.join(data_dir, "vggsound.csv"))
        all_examples = all_examples[all_examples["split"] == split]
        all_examples.loc[:, "augmented_id"] = (
            all_examples["id"]
            + "_"
            + all_examples["sec"].astype(str).apply(lambda x: x.zfill(6))
        )
        return all_examples.loc[all_examples["split"] == split, "augmented_id"].values

    def __init__(
        self,
        data_dir: str,
        num_examples: int = 100,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
    ):
        """Initialize the dataset.

        Args:
            data_dir (str): The path to the directory containing the downloaded data.
            num_examples (int, optional): The number of training examples to use.
                Defaults to 100.
            split (str, optional): The split to use. Defaults to "train".
            image_size (int, optional): The uniform image size to apply to the input images.
                Defaults to IMAGE_SIZE.
        """
        self.image_size = image_size
        self.image_processor = VaeImageProcessor()

        split_ids = self._get_split_ids(data_dir, split)

        image_root = os.path.join(data_dir, "frames")
        audio_root = os.path.join(data_dir, "audio")

        # get the ids of the downloaded images/audio that are in the right split
        downloaded_ids = [fname.split(".")[0] for fname in os.listdir(image_root)]
        valid_downloaded_ids = np.intersect1d(split_ids, downloaded_ids)
        print(f"there are {len(valid_downloaded_ids)} valid downloaded ids")
        print(f"using {num_examples} examples")
        id_sample = np.random.choice(
            valid_downloaded_ids,
            min(len(valid_downloaded_ids), num_examples),
            replace=False,
        )

        # get image and audio paths to load
        self.image_paths = [os.path.join(image_root, _id + ".jpg") for _id in id_sample]
        self.audio_paths = [os.path.join(audio_root, _id + ".wav") for _id in id_sample]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        audio_path = self.audio_paths[idx]
        base_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.image_processor.preprocess(
            image, self.image_size, self.image_size
        ).squeeze(1)

        # pylint: disable=no-member
        waveform, _ = torchaudio.load(audio_path)
        # pylint: enable=no-member
        return base_name, image, waveform.mean(dim=0)


def collate_fn(batch):
    """Collate function for the dataloader."""
    base_names, images, audios = zip(*batch)
    return (
        list(base_names),
        torch.concat(images, dim=0),
        torch.nn.utils.rnn.pad_sequence(audios, batch_first=True),
    )


# if __name__ == "__main__":
#     data_dir = "data"
#     num_examples = 100
#     split = "train"
#     image_size = IMAGE_SIZE

#     dataset = ImageAudioDataset(data_dir, num_examples, split, image_size)
#     data_loader = DataLoader(
#         dataset,
#         batch_size=8,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=os.cpu_count(),
#     )
#     for _, (names, img, aud) in enumerate(data_loader):
#         print(names)
#         print(img.shape)
#         print(img[:1, :1, ...])
#         print("------")
#         break
