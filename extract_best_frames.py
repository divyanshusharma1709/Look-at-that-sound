import os
import shutil
import sys

import clip
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from PIL import Image


def _extract_video_frame_sample(video_path: str, num_frames: int = 10):
    """Extracts and saves video frames to a temporary folder.

    Args:
        video_path (str): The path to the mp4 video file.
        frame_increment (int, optional): The number of frames to skip between each
            extracted frame. Defaults to 10.
    """
    # use opencv to read video frames and save them to frames_tmp
    # pylint: disable=no-member
    cap = cv2.VideoCapture(video_path)
    # pylint: enable=no-member
    if not cap.isOpened():
        return []
    # Get total number of frames in the video
    # pylint: disable=no-member
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # pylint: enable=no-member
    frame_idxs = np.random.permutation(total_frames)[:num_frames]
    frames = []
    for i in frame_idxs:
        # Set the current frame position of the video
        # pylint: disable=no-member
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # pylint: enable=no-member
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(torchvision.transforms.functional.to_pil_image(frame))
    return frames


def extract_best_frames(data_root: str):
    """Extracts the best frame from each video and saves it to the frames folder. Uses
    clip similarity between a text label and each frame to determine the best frame.

    Args:
        data_root (str): The root directory containing the downloaded data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, prepro = clip.load("ViT-B/32", device=device)

    os.makedirs(os.path.join(data_root, "frames"), exist_ok=True)

    df = pd.read_csv(os.path.join(data_root, "vggsound.csv"), header=0)
    # only use the videos that are locally available and skip any whose frames have already
    # been extracted
    downloaded_ids = {
        fname.split(".")[0] for fname in os.listdir(os.path.join(data_root, "video"))
    }
    already_extracted_ids = {
        fname.split(".")[0] for fname in os.listdir(os.path.join(data_root, "frames"))
    }
    df.loc[:, "augmented_id"] = (
        df["id"] + "_" + df["sec"].astype(str).apply(lambda x: x.zfill(6))
    )
    df = df[
        df["augmented_id"].isin(downloaded_ids)
        & ~df["augmented_id"].isin(already_extracted_ids)
    ]

    labels = df["label"].values
    augmented_ids = df["augmented_id"].values
    for i, aug_id in tqdm.tqdm(enumerate(augmented_ids), total=len(augmented_ids)):
        text = clip.tokenize(labels[i]).to(device)
        video_path = os.path.join(data_root, "video", aug_id + ".mp4")

        frames = _extract_video_frame_sample(video_path)

        if len(frames) > 0:
            images = torch.cat(
                [prepro(image).unsqueeze(0).to(device) for image in frames], dim=0
            )
            with torch.no_grad():
                logits_per_image, _ = clip_model(images, text)
                probs = logits_per_image.softmax(dim=0).cpu().numpy().T
                best_frame_idx = probs.argmax(axis=0)[0]
                torchvision.utils.save_image(
                    images[best_frame_idx],
                    os.path.join(data_root, f"frames/{aug_id}.jpg"),
                )


if __name__ == "__main__":
    extract_best_frames(sys.argv[1])
